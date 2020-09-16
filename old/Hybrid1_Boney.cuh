#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// -------------------------------------------- M A N D E L B O X   B O N E Y ------------------
// ---------------------------------------------------------------------------------------------

#define max_iter 120
#define bone make_float3(0.89f, 0.855f, 0.788f)

__device__ inline float DE(float3 z)
{
    //z.xz = rotate(z.xz, 0.1 * iTime);

    // menger params
    float3 MengerOffset = make_float3(1.0f);
    float MengerScale = 3.475f;//3.0;

    // mandelbox params
    float fixed_radius2 = 1.f;
    float min_radius2 = 0.f;
    float folding_limit = 1.65f;
    float MandelboxScale = 1.2f;// 1.2f + 0.2f * sin(0.5 * iTime);
    float3 MandelboxOffset = z;
    float dr = 1.0f;
    float r2;

    // shared
    int Iterations = 4;
    int n = 0;
    float both_scale = 1.0f;
    while (n < Iterations)
    {
        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs


        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs


        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs


        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs






        // menger step
        z = abs(z);
        if (z.x < z.y) {
            float temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            float temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            float temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - 1.0f); // same space transform as tetrahedron
        both_scale *= MengerScale;
        //dr *= MengerScale;

        if (z.z < -0.5f * MengerOffset.z * (MengerScale - 1.0f))
        {
            z.z += MengerOffset.z * (MengerScale - 1.0f);
        }

        // menger step
        z = abs(z);
        if (z.x < z.y) {
            float temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            float temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            float temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - 1.0f); // same space transform as tetrahedron
        both_scale *= MengerScale;
        //dr *= MengerScale;

        if (z.z < -0.5f * MengerOffset.z * (MengerScale - 1.0f))
        {
            z.z += MengerOffset.z * (MengerScale - 1.0f);
        }


        n++;
    }
    return length(z) / abs(both_scale + 1.f);

}


__device__ inline float intersect(float3 ro, float3 rd, float step_size, int& iter) {
    float res;
    float t = 0.f;
    iter = max_iter;
    for (int i = 0; i < max_iter; ++i) {
        float3 p = ro + rd * t;
        res = DE(p);
        if (res < 0.001f * t || res > 20.f) {
            iter = i;
            break;
        }
        t += res;
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
    float uvx = ((qx - 0.5f) * 2.0f) * (((float)WIDTH) / ((float)HEIGHT));  // range from -1 to 1
    float uvy = ((qy - 0.5f) * 2.0f); // range from -1 to 1


    // camera
    float3 lookat = _lookat;
    float3 ro = _position;
    float3 f = normalize(lookat - ro);
    float3 s = normalize(cross(f, make_float3(0.0f, 1.0f, 0.0f)));
    float3 u = normalize(cross(s, f));
    float3 rd = normalize(uvx * s + uvy * u + 2.8f * f);  // transform from view to world


    // background
    float3 bg = mix(bone * 0.5f, bone, smoothstep(-1.0f, 1.0f, uvy));
    float3 col = bg;
    float3 p = ro;
    int iter = 0;
    float t = intersect(ro, rd, EPS, iter);
    if (t > -0.5) {
        p = ro + t * rd;
        col = lighting(p, rd, iter, EPS);
        col = mix(col, bg, 1.0 - exp(-0.001 * t * t));
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



