#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// ------------------------------------------------ M A N D E L B O X   R Y U ------------------
// ---------------------------------------------------------------------------------------------

__device__ inline float TgladFold(float z, float Foldx) {
    return clamp(z, -Foldx, Foldx) * 2.0f - z;
}
__device__ void foldOct_(float3& z) {
    // BCDA,BDAC,BDCA, DABC,DBAC,DBCA - these combinations give same result as Mandelbulb3D
    if (z.x - z.z < 0.0f) z = make_float3(z.z, z.y, z.x);   // D
    if (z.x - z.y < 0.0f) z = make_float3(z.y, z.x, z.z);   // B 
    if (z.x + z.z < 0.0f) z = make_float3(-z.z, z.y, -z.x); // C
    if (z.x + z.y < 0.0f) z = make_float3(-z.y, -z.x, z.z); // A
}


__device__ inline void foldOct(float3& z, float plane1, float plane2, float plane3, float plane4 ) {
    bool test1, test2; float tmp;
    test1 = (z.x - z.z < plane1);
    test2 = !test1;
    tmp = z.z;
    z.z = z.z * test2 + z.x * test1;
    z.x = z.x * test2 + tmp * test1;
    test1 = (z.x - z.y < plane2);
    test2 = !test1;
    tmp = z.y;
    z.y = z.y * test2 + z.x * test1;
    z.x = z.x * test2 + tmp * test1;
    test1 = (z.x + z.z < plane3);
    test2 = !test1;
    tmp = z.z;
    z.z = z.z * test2 - z.x * test1;
    z.x = z.x * test2 - tmp * test1;
    test1 = (z.x + z.y < plane4);
    test2 = !test1;
    tmp = z.y;
    z.y = z.y * test2 - z.x * test1;
    z.x = z.x * test2 - tmp * test1;
}

__device__ inline float DE(float3 z, float plane1, float plane2, float plane3, float plane4,
    float Scale, float3 CScale, float Foldx, float Subx, float FoldMul, float ScaleOffset1, float ScaleOffset2) {
    float dr = 1.f; // CORR 
    bool test1, test2; float tmp; // locals 
    for (int n = 0; n < 17; ++n) {
        z = abs(z);
        
        test1 = (z.x - z.z < plane1);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x - z.y < plane2);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x + z.z < plane3);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;
        test1 = (z.x + z.y < plane4);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;


        z.x = z.x - Subx * (Scale - ScaleOffset1);
        z.x = clamp(z.x, -Foldx, Foldx) * FoldMul - z.x;
        //z.x = abs(z.x);//optional

        test1 = (z.x - z.z < plane1);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x - z.y < plane2);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x + z.z < plane3);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;
        test1 = (z.x + z.y < plane4);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;

        z -= CScale * (Scale - ScaleOffset2);
        z *= Scale;
        dr = dr * abs(Scale);//+ 1.; // CORR
    }
    return length(z) / abs(dr); // length is distance to shape, divided for correction!
}


__device__ inline float intersect(float3 ro, float3 rd, float step_size, float min_distance, 
    float plane1, float plane2, float plane3, float plane4,
    float Scale, float3 CScale, float Foldx, float Subx, float FoldMul, float ScaleOffset1, float ScaleOffset2)
{
    float res;
    float t = 0.f;
    float i = 0.f;
    for (; i < 120.f; i += 1.f)
    {
        float3 p = ro + rd * t;
        res = DE(p, plane1, plane2, plane3, plane4,Scale, CScale, Foldx, Subx, FoldMul, ScaleOffset1, ScaleOffset2);

        if (res < min_distance * t || res > 20.f) break;
        t += res * step_size;
    }

    //if (res > 20.f) t = -1.f;
    //return t;
    return i;
}


__global__ void kernel(uchar4* map, unsigned int iTime, float3 _position, float3 _lookat, float step_size, float EPS, float min_distance,
    float plane1, float plane2, float plane3, float plane4,
    float Scale, float3 CScale, float Foldx, float Subx, float FoldMul, float ScaleOffset1, float ScaleOffset2) {
    //  UV coords
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // ranges from 0 to 768
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // ranges from 0 to 512
    int idx = ix + iy * WIDTH;//blockDim.x * gridDim.x; 
    if (idx < 522753) // render top half of screen
    {
        float qx = ((float)ix) / WIDTH; // ranges from 0 to 1
        float qy = ((float)iy) / (HEIGHT - HEIGHT_OFFSET); // ranges from 0 to 1
        float uvx = ((qx - 0.5f) * 2.0f) * (WIDTH / (HEIGHT - HEIGHT_OFFSET));  // range from -1 to 1
        float uvy = ((qy - 0.5f) * 2.0f); // range from -1 to 1


        // camera
        //float stime = sin(iTime * 0.1f); float ctime = cos(iTime * 0.1f); float time = iTime * 0.01f;
        float3 ro = _position;
        float3 lookat = _lookat;
        float3 f = normalize(lookat - ro); //   F O R W A R D 
        float3 r = cross(f, make_float3(0.0f, 1.0f, 0.0f)); //   S I D E ? TODO: left or right, right uses diff order in cross prod
        float3 u = cross(r, f); // UP ? TODO: also uses diff order in cross prod
        float3 rd = normalize(uvx * r + uvy * u + 2.8f * f);  // transform from view to world


        float t = intersect(ro, rd, step_size, min_distance, plane1, plane2, plane3, plane4, Scale, CScale, Foldx, Subx, FoldMul, ScaleOffset1, ScaleOffset2);
        t *= 5.f;
        map[idx].x = (unsigned char)t;
        map[idx].y = (unsigned char)0;
        map[idx].z = (unsigned char)0;
        map[idx].w = (unsigned char)255;
    }
    
}


float plane1 = 0.f;
float plane2 = 0.f;
float plane3 = 0.f;
float plane4 = 0.f;
float Scale = 2.4f;
float3 CScale = make_float3(-0.75f, 0.25f, 0.25f);
float Foldx = 1.f;
float Subx = 2.f;
float FoldMul = 2.f;
float ScaleOffset1 = 1.f; 
float ScaleOffset2 = 1.f;

extern "C" void Mandelbox(Camera * camera_ptr)
{
    static unsigned int iTime = 0;
    iTime++;

    uchar4* dev_map;
    dim3 threads(8, 8);
    dim3 grids(WIDTH / 8, HEIGHT / 8); // 96 x 64

    gpuErrchk(cudaGraphicsMapResources(1, &cudapbo, NULL));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));
    kernel << <grids, threads >> > (dev_map, iTime, camera_ptr->position, camera_ptr->lookat, camera_ptr->step_size, camera_ptr->EPS, camera_ptr->min_distance, plane1, plane2, plane3, plane4, Scale, CScale, Foldx, Subx, FoldMul, ScaleOffset1, ScaleOffset2);
    //gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));
}


extern "C" void RenderOptions()
{
    // ImGui 
    ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    ImGui::Begin("Mandelbox-------------------");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::SliderFloat("Plane 1", &plane1, -5.f, 5.f);
    ImGui::SliderFloat("Plane 2", &plane2, -5.f, 5.f);
    ImGui::SliderFloat("Plane 3", &plane3, -5.f, 5.f);
    ImGui::SliderFloat("Plane 4", &plane4, -5.f, 5.f);
    ImGui::SliderFloat("Scale", &Scale, -5.f, 5.f);
    ImGui::SliderFloat("CScale X", &CScale.x, -5.f, 5.f);
    ImGui::SliderFloat("CScale Y", &CScale.y, -5.f, 5.f);
    ImGui::SliderFloat("CScale Z", &CScale.z, -5.f, 5.f);
    ImGui::SliderFloat("Fold X", &Foldx, -5.f, 5.f);
    ImGui::SliderFloat("Sub X", &Subx, -5.f, 5.f);
    ImGui::SliderFloat("Fold Mul", &FoldMul, -5.f, 5.f);
    ImGui::SliderFloat("Scale Offset 1", &ScaleOffset1, -5.f, 5.f);
    ImGui::SliderFloat("Scale Offset 2", &ScaleOffset2, -5.f, 5.f);

    //ImGui::SliderFloat("base_color R", &camera.base_color.x, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::End();
}