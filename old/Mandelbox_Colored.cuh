#include "../Common.cuh"
#include "../Camera.h"
//#include "Mandelbox_Ryu.cuh"


// ---------------------------------------- C O L O R E D   M A N D E L B O X ------------------
// ---------------------------------------------------------------------------------------------

cudaGraphicsResource* cudapbo;



__global__ void kernel(uchar4* map, unsigned int iTime, float3 _position, float3 _lookat, float step_size, float EPS) {
    //  UV coords
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // ranges from 0 to 768
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // ranges from 0 to 512
    int idx = ix + iy * WIDTH;//blockDim.x * gridDim.x; 
    float qx = ((float)ix) / WIDTH; // ranges from 0 to 1
    float qy = ((float)iy) / HEIGHT; // ranges from 0 to 1
    float uvx = ((qx - 0.5f) * 2.0f) * (WIDTH / HEIGHT);  // range from -1 to 1
    float uvy = ((qy - 0.5f) * 2.0f); // range from -1 to 1


    float3 col;
    float3 r = normalize(make_float3(uvx, uvy, 1.f));
    float3 p = _lookat;// make_float3(-.44f, .11f, -10.f + iTime / 2.f);
    for (float i = .0f; i < 99.f; i++)
    {
        float4 o = make_float4(p.x, p.y, p.z, 1.f);
        float4 q = o;
        for (float i = 0.f; i < 9.f; i++) {
            float3 temp = clamp(make_float3(o.x, o.y, o.z), -1.f, 1.f) * 2.f;
            o.x = temp.x - o.x;
            o.y = temp.y - o.y;
            o.z = temp.z - o.z;
            float3 t2 = make_float3(o.x, o.y, o.z);
            float hmm = max(.25f / dot(t2, t2), .25f);
            o = o * clamp(hmm, 0.f, 1.f) * make_float4(11.2f, 11.2f, 11.2f, 11.2f) + q;
        }
        float d = (length(make_float3(o.x, o.y, o.z)) - 1.f) / o.w - 5e-4f;
        if (d < 5e-4f) { break; }
        p += r * d;
        col = (1.f - i / 50.f - normalize(make_float3(o.x, o.y, o.z)) * .25f);
    }

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
