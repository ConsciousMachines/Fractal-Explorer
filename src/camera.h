#ifndef CAMERA_H
#define CAMERA_H

#include <helper_math.h>
#include "definitions.h"
// #include "file_ops.h"

// camera keeps track of what we see and when to render it.
class Camera 
{
public:
    // ray origin, and lookat vecs
    float3 position;
    float3 lookat;

    // struct of 20 float parameters to pass to CUDA kernel
    Params params;

    // tell main loop if we should quit
    bool should_render = true;

    // attached to glfw keypress callback 
    void keypress_callback(int, int);

    // called every frame to change position & render
    void move();

    // launch the CUDA kernel
    void launch_kernel();

    // function to render/reset ImGui for specific fractal
    void (*render_options)(Camera&);
    void (*reset_options)(Camera&);

    // GPU memory pointer
    uchar4* dst;

    // file manager to save stuff
    // file_manager fm;

    Camera();
    struct cudaGraphicsResource* cuda_pbo_resource; // needed to take photo
private:
    // variables to keep track of how we are moving
    float MOV_AMT = 0.01f;  // how slowly we are moving (zooming)
    float ROT_AMT = 0.03f; // how fast we rotate
    int is_moving = 0;     // flag signaling to change position and render
    enum MOVE_TYPE { MOVF, MOVB, MOVU, MOVD, MOVR, MOVL, ROTR, ROTL };
    MOVE_TYPE move_type;
};

#endif