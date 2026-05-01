#include <optix.h>
#include <optix_micromap.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <prayground/math/vec.h>
#include <prayground/optix/util.h>
#include <prayground/optix/omm.h>

namespace prayground {
    namespace {
        constexpr float kOpacityCutoff = 0.0f;

        INLINE DEVICE bool isTransparentAlpha(float alpha)
        {
            return alpha <= kOpacityCutoff;
        }
    }

    INLINE DEVICE float signedTriangleArea(Vec2f a, Vec2f b, Vec2f c) {
        return cross(b - a, c - a) / 2.0f;
    }

    DEVICE uint8_t evaluateTransparencyInSingleMicroTriangle(
        OptixOpacityMicromapFormat format,
        Vec2f uv0, Vec2f uv1, Vec2f uv2,
        OpacityMicromap::MicroBarycentrics bc, 
        Vec2i tex_size,
        cudaTextureObject_t texture)
    {
        Vec2f step = Vec2f(1.0f) / Vec2f(tex_size);

        const Vec2f bary0 = barycentricInterop(uv0, uv1, uv2, bc.uv0);
        const Vec2f bary1 = barycentricInterop(uv0, uv1, uv2, bc.uv1);
        const Vec2f bary2 = barycentricInterop(uv0, uv1, uv2, bc.uv2);

        Vec2f corner_min = Vec2f{
            fminf(fminf(bary0.x(), bary1.x()), bary2.x()), 
            fminf(fminf(bary0.y(), bary1.y()), bary2.y())
        };
        Vec2f corner_max = Vec2f{
            fmaxf(fmaxf(bary0.x(), bary1.x()), bary2.x()), 
            fmaxf(fmaxf(bary0.y(), bary1.y()), bary2.y())
        };

        auto corner_size = corner_max - corner_min;
        int est_pixels_x = max(1, static_cast<int>(ceilf(corner_size.x() / step.x())));
        int est_pixels_y = max(1, static_cast<int>(ceilf(corner_size.y() / step.y())));
        int64_t est_total_pixels = static_cast<int64_t>(est_pixels_x) * est_pixels_y;

        auto classify = [&](float alpha) -> uint8_t {
            if (!isTransparentAlpha(alpha)) {
                if (format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE)
                    return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
                return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            }
            return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
        };

        float4 c0 = tex2D<float4>(texture, bary0.x(), bary0.y());
        float4 c1 = tex2D<float4>(texture, bary1.x(), bary1.y());
        float4 c2 = tex2D<float4>(texture, bary2.x(), bary2.y());
        uint8_t state0 = classify(c0.w);
        if (state0 != OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT) return state0;
        uint8_t state1 = classify(c1.w);
        if (state1 != OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT) return state1;
        uint8_t state2 = classify(c2.w);
        if (state2 != OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT) return state2;

        if (est_total_pixels > 64) {
            const int SAMPLES = 4;
            for (int iy = 0; iy < SAMPLES; ++iy) {
                for (int ix = 0; ix < SAMPLES; ++ix) {
                    float sx = corner_min.x() + (corner_size.x() * (ix + 0.5f) / SAMPLES);
                    float sy = corner_min.y() + (corner_size.y() * (iy + 0.5f) / SAMPLES);
                    Vec2f uv(sx, sy);
                    auto area01 = signedTriangleArea(uv, bary0, bary1);
                    auto area12 = signedTriangleArea(uv, bary1, bary2);
                    auto area20 = signedTriangleArea(uv, bary2, bary0);
                    if ((area01 >= 0 && area12 >= 0 && area20 >= 0) || (area01 <= 0 && area12 <= 0 && area20 <= 0)) {
                        float4 color = tex2D<float4>(texture, sx, sy);
                        if (!isTransparentAlpha(color.w)) {
                            if (format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE)
                                return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
                            return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
                        }
                    }
                }
            }
            return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
        }

        for (float y = corner_min.y(); y <= corner_max.y(); y += step.y()) {
            for (float x = corner_min.x(); x <= corner_max.x(); x += step.x()) {
                Vec2f uv(x, y);
                auto area01 = signedTriangleArea(uv, bary0, bary1);
                auto area12 = signedTriangleArea(uv, bary1, bary2);
                auto area20 = signedTriangleArea(uv, bary2, bary0);
                if ((area01 >= 0 && area12 >= 0 && area20 >= 0) || (area01 <= 0 && area12 <= 0 && area20 <= 0)) {
                    float4 color = tex2D<float4>(texture, x, y);
                    if (!isTransparentAlpha(color.w)) {
                        if (format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE)
                            return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
                        return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
                    }
                }
            }
        }

        return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
    }

    extern "C" GLOBAL void generateOpacityMap(
        uint32_t* d_out_omm_data, 
        int32_t subdivision_level,
        int32_t num_faces,
        OptixOpacityMicromapFormat format,
        Vec2i tex_size, 
        const Vec2f* d_texcoords, const Vec3i* d_faces,
        cudaTextureObject_t texture) 
    {
        const int num_micro_triangles = 1 << (subdivision_level * 2);
        size_t num_elems_per_face = max((num_micro_triangles / 32 * format), 1);

        // Compute a linear global thread id across a 2D grid of blocks.
        int64_t linear_block_idx = static_cast<int64_t>(blockIdx.x) + static_cast<int64_t>(blockIdx.y) * static_cast<int64_t>(gridDim.x);
        int64_t global_thread_id = linear_block_idx * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
        if (global_thread_id >= num_micro_triangles * num_faces)
            return;

        int64_t face_idx = global_thread_id / num_micro_triangles;
        int64_t micro_tri_idx = global_thread_id % num_micro_triangles;
            
        int num_states_per_elem = 32 / format;
        
        const Vec2f uv0 = d_texcoords[d_faces[face_idx].x()];
        const Vec2f uv1 = d_texcoords[d_faces[face_idx].y()];
        const Vec2f uv2 = d_texcoords[d_faces[face_idx].z()];

        float2 bary0, bary1, bary2;
        optixMicromapIndexToBaseBarycentrics(micro_tri_idx, subdivision_level, bary0, bary1, bary2);
        
        OpacityMicromap::MicroBarycentrics bc{bary0, bary1, bary2};

        uint8_t state = evaluateTransparencyInSingleMicroTriangle(format, uv0, uv1, uv2, bc, tex_size, texture);
        int64_t index = global_thread_id / num_states_per_elem;
        uint32_t* address = &d_out_omm_data[index];
        const uint32_t shift = static_cast<uint32_t>((micro_tri_idx % num_states_per_elem) * format);
        atomicOr(address, static_cast<uint32_t>(state) << shift);
    }

    extern "C" HOST void evaluateSingleOpacityTexture(
        uint32_t * d_out_omm_data, // GPU pointer to the output opacity map
        int32_t subdivision_level,
        int32_t num_faces,
        OptixOpacityMicromapFormat format,
        Vec2i tex_size,
        const Vec2f* d_texcoords, const Vec3i* d_faces,
        cudaTextureObject_t texture
    ) {
        // Configure a safe launch: use a conservative threads-per-block and compute
        // a 2D grid that fits within CUDA limits.
        constexpr int NUM_MAX_THREADS = 256;
        // Max blocks per dimension (older GPUs have 65535). Use 65535 to be safe.
        constexpr int NUM_MAX_BLOCKS = 65535;

        const int num_micro_triangles = 1 << (subdivision_level * 2);
        const int threads_per_block = min(num_micro_triangles, NUM_MAX_THREADS);
        dim3 threads(threads_per_block, 1, 1);

        const int64_t total_threads = static_cast<int64_t>(num_micro_triangles) * static_cast<int64_t>(num_faces);
        const int64_t total_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

        int block_size_x = static_cast<int>(min<int64_t>(total_blocks, NUM_MAX_BLOCKS));
        int block_size_y = static_cast<int>((total_blocks + block_size_x - 1) / block_size_x);
        dim3 grid(block_size_x, block_size_y, 1);

        generateOpacityMap<<<grid, threads>>>(d_out_omm_data, subdivision_level, num_faces, format, tex_size, d_texcoords, d_faces, texture);
    }
}