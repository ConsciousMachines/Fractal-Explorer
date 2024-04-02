#pragma once 

#define INTERACTIVE // this is for no OpenGL - just a clean CUDA thing that outputs an image. 
#define RENDER_EVERY_FRAME // for rendering every frame, seeing changes in real time. gets hot.


#ifdef INTERACTIVE
#define WIDTH 1024//512//1024//768
#define HEIGHT (512 + 256)//512//256//512
#define HEIGHT_OFFSET 256
#else
#define WIDTH 1920//3840//1920
#define HEIGHT 1080//2160//1080
#define HEIGHT_OFFSET 0
#endif

#define FILENAME "params.txt"
#define INITFILENAME "init.txt"


#include <stdlib.h>
#include <iostream>
#include <GL/glew.h> // GL 
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h> // CUDA
#include <device_launch_parameters.h>
#include "helper_math.h" // GLSL-like math 
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"


#define RANGE(x, y) for (int x = 0; x < y; x++)
#define PRINT(x) std::cout << x << "\n";
#define END std::cin.get(); return 420;


#define M_PI 3.14159265359f
#define degreesToRadians(angleDegrees) (angleDegrees * M_PI / 180.0f)
#define gpuErrchk(ans) ans
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
//    if (code != cudaSuccess) { 
//        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//        if (abort) exit(code);}}


__device__ inline float mix(float v1, float v2, float a);
__device__ inline float3 abs(float3 a);
// https://developer.download.nvidia.com/cg/pow.html
__device__ inline float3 pow(float3 x, float3 y);
// https://www.reddit.com/r/opengl/comments/6nghtj/glsl_mix_implementation_incorrect/
__device__ inline float3 mix(float3 v1, float3 v2, float a);


/*
class quaternion {
    float x, y, z, w;
    float length() { return sqrt(this->x * this->x + this->y * this->y + this->z * this->z + this->w * this->w); }
    void normalize() { float L = length(); this->x /= L; this->y /= L; this->z /= L; this->w /= L; }
    void conjugate() { this->x = -this->x; this->y = -this->y; this->z = -this->z; }
    static quaternion mult(quaternion A, quaternion B) { quaternion C;
    C.x = A.w * B.x + A.x * B.w + A.y * B.z - A.z * B.y;
    C.y = A.w * B.y - A.x * B.z + A.y * B.w + A.z * B.x;
    C.z = A.w * B.z + A.x * B.y - A.y * B.x + A.z * B.w;
    C.w = A.w * B.w - A.x * B.x - A.y * B.y - A.z * B.z; return C; }
};
*/




/*
#ifdef REFLECTIONS
// REFLECTIONS
float3 p_ = ro + t * rd;
float3 normal_ = normal(p_, EPS, params); // the normal from the hit point
float iter2 = 0.f;
float3 col2 = bg;
float d2 = intersect(p_, normal_, step_size, min_distance, EPS, iter2, params); // ray march from that point to scene
float3 p2_ = p_ + d2 * normal_; // the ray from surface that hit scene
if (d2 > -0.5f) {
    col2 = lighting(p2_, normal_, iter2, EPS, params);
    col2 = mix(col2, bg, 1.0f - exp(-0.001f * d2 * d2));

    // final color
    col = 0.5 * col + 0.5 * col2;
}
#endif
*/


//#include "Fractals/Mandelbox_Ryu_Original.cuh" // DO NOT CHANGE - reference for lighting model!!!
//#include "Fractals/Mandelbox_Ryu.cuh" // first, nothing special
//#include "Fractals/Mandelbox_Colored.cuh" // nice but cant tell what the code does
//#include "Fractals/Mandelbox_Ryu_Full.cuh" // full Ryu with my crappy AO - i like (brighter fractal + blue tint)
//#include "Fractals/Mandelbox_RyuAO.cuh" // Ryu with AO from boney. come back to this when i understand lighting
//#include "Fractals/Mandelbox_BoneyGLOW.cuh" // pretty solid glow, but i still had to multiply to use with AO. 
//#include "Fractals/Mandelbox_BoneyDOUBLE.cuh"
//#include "Fractals/Mandelbox_BoneyGLO_REFL.cuh"
//#include "Fractals/OctKoch.cuh" // slow 
//#include "Fractals/OctKoch_Color.cuh" // slow 
//#include "Fractals/OctKoch_Debug.cuh"  // looks like an evil castle!
//#include "Fractals/MengerSmt.cuh" // worked on first try :D
//#include "Fractals/Hybrid1.cuh" // can actually be OK with the right parameters if we ignore the background
//#include "Fractals/Mandelbox_Rainbow.cuh" // hmm
//#include "Fractals/Hybrid1_Boney.cuh" // absolutely terrible!
//#include "Fractals/Renderer.cuh" // i give up with lights for now 
//#include "Fractals/Mandelbox_Boney.cuh" // very nice AO - Reaktor logo 
//#include "Fractals/OctKoch_GUI.cuh"
//#include "Fractals/Mandelbox_Boney_GUI.cuh" // new favorite



// Unified Version 
//uchar4* u_fb; // unified frame buffer
//checkCudaErrors(cudaMallocManaged((void**)&u_fb, fb_size));
//kernel << <grids, threads >> > (u_fb, iTime, camera.position, camera.lookat, camera.step_size, camera.EPS);
//checkCudaErrors(cudaGetLastError());
//checkCudaErrors(cudaDeviceSynchronize());
//int SUCCESS = stbi_write_bmp("C:\\Users\\pwnag\\Desktop\\test.bmp", WIDTH, HEIGHT, 4, u_fb);
//checkCudaErrors(cudaFree(u_fb));