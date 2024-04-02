#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// -------------------------------------------- M A N D E L B O X   B O N E Y ------------------
// ---------------------------------------------------------------------------------------------

#define max_iter 120
#define bone make_float3(0.89f, 0.855f, 0.788f)




__device__ inline float DE(float3 z, float fold, float fixed_radius2, float MinR2, float scale, float box_mult) 
{
    float3 offset = z;

    float dr = 1.0f;
    for (int n = 0; n < 15; ++n) {
        z = clamp(z, -fold, fold) * box_mult - z; // box fold
        
        float r2 = dot(z, z); // sphere fold 
        if (r2 < MinR2) {
            float temp = (fixed_radius2 / MinR2);
            z *= temp;
            dr *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
        }
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0f;
    }
    float r = length(z);
    return r / abs(dr);
}

__device__ inline float intersect(float3 ro, float3 rd, float step_size, int& iter, float EPS,
    float fold, float fixed_radius2, float MinR2, float scale, float box_mult, float min_distance) {
    float res;
    float t = 0.f;
    iter = max_iter;
    for (int i = 0; i < max_iter; ++i) {
        float3 p = ro + rd * t;
        res = DE(p, fold, fixed_radius2, MinR2, scale, box_mult);
        if (res < min_distance * t || res > 20.f) {
            iter = i;
            break;
        }
        t += res * step_size;
    }
    if (res > 20.f) t = -1.f;
    return t;
}

__device__ inline float ambientOcclusion(float3 p, float3 n, float fold, float fixed_radius2, float MinR2, float scale, float box_mult) {
    float stepSize = 0.012f;
    float t = stepSize;
    float oc = 0.0f;
    for (int i = 0; i < 12; i++) {
        float d = DE(p + n * t, fold, fixed_radius2, MinR2, scale, box_mult);
        oc += t - d;
        t += stepSize;
    }
    return clamp(oc, 0.0f, 1.0f);
}

__device__ inline float3 normal(float3 p, float EPS, float fold, float fixed_radius2, float MinR2, float scale, float box_mult) {

    float3 e0 = make_float3(EPS, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, EPS, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, EPS);
    float3 n = normalize(make_float3(
        DE(p + e0, fold, fixed_radius2, MinR2, scale, box_mult) - DE(p - e0, fold, fixed_radius2, MinR2, scale, box_mult),
        DE(p + e1, fold, fixed_radius2, MinR2, scale, box_mult) - DE(p - e1, fold, fixed_radius2, MinR2, scale, box_mult),
        DE(p + e2, fold, fixed_radius2, MinR2, scale, box_mult) - DE(p - e2, fold, fixed_radius2, MinR2, scale, box_mult)));
    return n;
}

__device__ inline float3 lighting(float3 p, float3 rd, int iter, float EPS, float fold, float fixed_radius2, float MinR2, float scale, float box_mult) {
    float3 n = normal(p, EPS, fold, fixed_radius2, MinR2, scale, box_mult);
    float fake = float(iter) / float(max_iter);
    float fakeAmb = exp(-fake * fake * 9.0f);
    float amb = ambientOcclusion(p, n, fold, fixed_radius2, MinR2, scale, box_mult);
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

__global__ void kernel(uchar4* map, unsigned int iTime, float3 _position, float3 _lookat, float step_size, float EPS,
    float fold, float fixed_radius2, float MinR2, float scale, float box_mult, float min_distance) {
    //  UV coords
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // ranges from 0 to 768
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // ranges from 0 to 512
    int idx = ix + iy * WIDTH;//blockDim.x * gridDim.x; 
    if (idx < ((WIDTH-1)*(HEIGHT-1))) // render top half of screen
    {
        float qx = ((float)ix) / WIDTH; // ranges from 0 to 1
        float qy = ((float)iy) / (HEIGHT - HEIGHT_OFFSET); // ranges from 0 to 1
        float uvx = ((qx - 0.5f) * 2.0f) * (WIDTH / (HEIGHT - HEIGHT_OFFSET));  // range from -1 to 1
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
        float t = intersect(ro, rd, step_size, iter, EPS, fold, fixed_radius2, MinR2, scale, box_mult, min_distance);
        if (t > -0.5) {
            p = ro + t * rd;
            col = lighting(p, rd, iter, EPS, fold, fixed_radius2, MinR2, scale, box_mult);
            col = mix(col, bg, 1.0 - exp(-0.001 * t * t));
        }
        col = post(col, make_float2(qx, qy));
        col = clamp(col, 0.f, 1.f); // sev genius input. there is a bit of overflow somewhere but this fixes it.
        map[idx].x = (unsigned char)(255.0f * col.x);
        map[idx].y = (unsigned char)(255.0f * col.y);
        map[idx].z = (unsigned char)(255.0f * col.z);
        map[idx].w = (unsigned char)255;
    }
}


float scale = -2.8f;
float fixed_radius2 = 1.9f;
float MinR2 = 0.1f;
float fold = 1.0f;
float box_mult = 2.f;


extern "C" void Mandelbox(Camera * camera_ptr)
{
    //static int i = 0;
    //printf("RENDERING frame %i\n", i++);
    static unsigned int iTime = 0;
    iTime++;

    uchar4* dev_map;
    dim3 threads(8, 8);
    dim3 grids(WIDTH / 8, HEIGHT / 8); // 96 x 64

    gpuErrchk(cudaGraphicsMapResources(1, &cudapbo, NULL));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));
    kernel << <grids, threads >> > (dev_map, iTime, camera_ptr->position, camera_ptr->lookat, camera_ptr->step_size, camera_ptr->EPS, fold, fixed_radius2, MinR2, scale, box_mult, camera_ptr->min_distance);
    //gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));
}



extern "C" void RenderOptions()
{
    // ImGui 
    ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    ImGui::Begin("Mandelbox-------------------");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::SliderFloat("Scale", &scale, -5.f, 5.f);
    ImGui::SliderFloat("Fixed Radius", &fixed_radius2, -5.f, 5.f);
    ImGui::SliderFloat("Min R", &MinR2, -5.f, 5.f);
    ImGui::SliderFloat("Fold", &fold, -5.f, 5.f);
    ImGui::SliderFloat("Box Mult", &box_mult, -5.f, 5.f);
    //ImGui::SliderFloat("base_color R", &camera.base_color.x, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::End();
}

