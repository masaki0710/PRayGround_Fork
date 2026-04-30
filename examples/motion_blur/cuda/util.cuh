#pragma once 

#include <prayground/prayground.h>
#include "../params.h"

namespace prayground {

    extern "C" {
        __constant__ LaunchParams params;
    }

    struct SurfaceInteraction
    {
        float3 p;
        float3 n;
        float3 albedo;
        float3 shading_val;
        float2 uv;
    };

    INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
    {
        const unsigned int u0 = optixGetPayload_0();
        const unsigned int u1 = optixGetPayload_1();
        return reinterpret_cast<SurfaceInteraction*>( unpackPointer(u0, u1) ); 
    }

    // -------------------------------------------------------------------------------
    INLINE DEVICE void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float                  ray_time, 
        SurfaceInteraction*    si
    )
    {
        unsigned int u0, u1;
        packPointer( si, u0, u1 );
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            ray_time,          // ray time
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,        
            1,           
            0,        
            u0, u1 );	
    }

} // namespace prayground