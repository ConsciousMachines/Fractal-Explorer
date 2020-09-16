#include "Common.cuh"





__device__ inline float mix(float v1, float v2, float a)
{
    return v1 * (1.f - a) + v2 * a;
}
__device__ inline float3 abs(float3 a)
{
    return make_float3(abs(a.x), abs(a.y), abs(a.z));
}

// https://developer.download.nvidia.com/cg/pow.html
__device__ inline float3 pow(float3 x, float3 y)
{
    float3 rv;
    rv.x = pow(x.x, y.x);//exp(x.x * log(y.x));
    rv.y = pow(x.y, y.y);//exp(x.y * log(y.y));
    rv.z = pow(x.z, y.z);//exp(x.z * log(y.z));
    return rv;
}


// https://www.reddit.com/r/opengl/comments/6nghtj/glsl_mix_implementation_incorrect/
__device__ inline float3 mix(float3 v1, float3 v2, float a)
{
    float3 result;
    result.x = v1.x * (1.f - a) + v2.x * a;
    result.y = v1.y * (1.f - a) + v2.y * a;
    result.z = v1.z * (1.f - a) + v2.z * a;
    return result;
}

