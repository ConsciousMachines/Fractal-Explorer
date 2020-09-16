#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// -------------------------------------------- M A N D E L B O X   B O N E Y ------------------
// ---------------------------------------------------------------------------------------------

#define max_iter 120
#define bone make_float3(0.89f, 0.855f, 0.788f)


// Represents the maximum RGB color shift that can result from glow.
// Recommended Range : All values between - 1.0 and 1.0
#define GLOW_COLOR_DELTA make_float3(0.4f, 1.f, 1.f)

// The sharpness of the glow.
// Recommended Range : 1.0 to 100.0
#define GLOW_SHARPNESS 10.1f 


__device__ inline void sphere_fold(float3& z, float& dz) {
    float fixed_radius2 = 1.9f;
    float min_radius2 = 0.1f;
    float r2 = dot(z, z);
    if (r2 < min_radius2) {
        float temp = (fixed_radius2 / min_radius2);
        z *= temp;
        dz *= temp;
    }
    else if (r2 < fixed_radius2) {
        float temp = (fixed_radius2 / r2);
        z *= temp;
        dz *= temp;
    }
}

__device__ inline void box_fold(float3& z, float& dz) {
    float folding_limit = 1.0f;
    z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
}

__device__ inline float DE(float3 z) {
    float scale = -2.8f;
    float3 offset = z;
    float dr = 1.0f;
    for (int n = 0; n < 15; ++n) {
        box_fold(z, dr);
        sphere_fold(z, dr);
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0f;
        //scale = -2.8 - 0.2 * stime;
    }
    float r = length(z);
    return r / abs(dr);
}

__device__ inline float intersect(float3 ro, float3 rd, float step_size, int& iter, float& min_d) {
    float res;
    float t = 0.0011f; // FOR REFLECTION!
    iter = max_iter;
    for (int i = 0; i < max_iter; ++i) {
        float3 p = ro + rd * t;
        res = DE(p);
        if (res < 0.001f * t || res > 20.f) {
            iter = i;
            break;
        }
        t += res;
        min_d = min(min_d, GLOW_SHARPNESS * res / t);
    }
    if (res > 20.f) t = -1.f;
    return t;
}

__device__ inline float ambientOcclusion(float3 p, float3 n) {
    float stepSize = 0.012f;
    float t = stepSize;
    float oc = 0.0f;
    for (int i = 0; i < 12; i++) {
        float d = DE(p + n * t);
        oc += t - d;
        t += stepSize;
    }
    return clamp(oc, 0.0f, 1.0f);
}

__device__ inline float3 normal(float3 p, float EPS) {

    float3 e0 = make_float3(EPS, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, EPS, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, EPS);
    float3 n = normalize(make_float3(
        DE(p + e0) - DE(p - e0),
        DE(p + e1) - DE(p - e1),
        DE(p + e2) - DE(p - e2)));
    return n;
}

__device__ inline float3 lighting(float3 p, float3 rd, int iter, float EPS) {
    float3 n = normal(p, EPS);
    float fake = float(iter) / float(max_iter);
    float fakeAmb = exp(-fake * fake * 9.0f);
    float amb = ambientOcclusion(p, n);
    float3 col = make_float3(mix(1.0f, 0.125f, pow(amb, 3.0f))) * make_float3(fakeAmb) * bone;

    return col;
}

__device__ inline float3 post(float3 col, float2 q) {
    col = pow(clamp(col, 0.0f, 1.0f), make_float3(0.45f));
    col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);  // contrast
    col = mix(col, make_float3(dot(col, make_float3(0.33f))), -0.5f);  // satuation
    col *= 0.5f + 0.5f * pow(19.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.7f);  // vigneting
    return col;
}

__global__ void kernel(uchar4* map, unsigned int iTime, float3 _position, float3 _lookat, float step_size, float EPS) {
    //  UV coords
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // ranges from 0 to 768
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // ranges from 0 to 512
    int idx = ix + iy * WIDTH;//blockDim.x * gridDim.x; 
    float qx = ((float)ix) / WIDTH; // ranges from 0 to 1
    float qy = ((float)iy) / HEIGHT; // ranges from 0 to 1
    float uvx = ((qx - 0.5f) * 2.0f) * (WIDTH / HEIGHT);  // range from -1 to 1
    float uvy = ((qy - 0.5f) * 2.0f); // range from -1 to 1


    // camera
    float3 lookat = _lookat;
    float3 ro = _position;
    float3 f = normalize(lookat - ro);
    float3 s = normalize(cross(f, make_float3(0.0f, 1.0f, 0.0f)));
    float3 u = normalize(cross(s, f));
    float3 rd = normalize(uvx * s + uvy * u + 2.8f * f);  // transform from view to world


    // background
    float3 bg = make_float3(0.f);// mix(bone * 0.5f, bone, smoothstep(-1.0f, 1.0f, uvy)); // BACKGROUND REFLECTS TOO WTF
    float3 col = bg;
    float3 p = ro;
    int iter = 0;
    float m = 1.f; // GLOW 
    float t = intersect(ro, rd, EPS, iter, m);
    if (t > -0.5f) {
        p = ro + t * rd;
        col = lighting(p, rd, iter, EPS);
        col = mix(col, bg, 1.0f - exp(-0.001f * t * t));
    }
    /*
    */
    auto glow = (1.0f - m) * (1.0f - m) * GLOW_COLOR_DELTA; // GLOW 
    glow = pow(glow, make_float3(2.f));// make lows lower and highs higher
    col *= glow; // the AO and glow seemed to be fighting. kept the darks by multiplying again 


    // REFLECTIONS
    float3 p_ = ro + t * rd;
    float3 normal_ = normal(p_, EPS); // the normal from the hit point
    int iter2 = 0;
    float m2 = 1.f; 
    float3 col2 = bg;
    float d2 = intersect(p_, normal_, EPS, iter2, m2); // ray march from that point to scene
    float3 p2_ = p_ + d2 * normal_; // the ray from surface that hit scene
    if (d2 > -0.5f) {
        col2 = lighting(p2_, normal_, iter2, EPS);
        col2 = mix(col2, bg, 1.0f - exp(-0.001f * d2 * d2));

        // final color 
        col = 0.5 * col + 0.5 * col2;
    }

    col = post(col, make_float2(qx, qy));
    col = clamp(col, 0.f, 1.f); // sev genius input. there is a bit of overflow somewhere but this fixes it.
    map[idx].x = (unsigned char)(255.0f * col.x);//(unsigned char)(255.0f * uvx);//ix / 2;
    map[idx].y = (unsigned char)(255.0f * col.y);//(unsigned char)(255.0f * uvy);//iy / 2;
    map[idx].z = (unsigned char)(255.0f * col.z);//(unsigned char)iTime;
    map[idx].w = (unsigned char)255;
}

extern "C" void Mandelbox(Camera * camera_ptr)
{
    static unsigned int iTime = 0;
    iTime++;

    uchar4* dev_map;
    dim3 threads(8, 8);
    dim3 grids(WIDTH / 8, HEIGHT / 8); // 96 x 64

    gpuErrchk(cudaGraphicsMapResources(1, &cudapbo, NULL));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));
    kernel << <grids, threads >> > (dev_map, iTime, camera_ptr->position, camera_ptr->lookat, camera_ptr->step_size, camera_ptr->EPS);
    //gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));
}



