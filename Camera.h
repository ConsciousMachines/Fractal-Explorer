#pragma once 

#include "Common.cuh"
#include <string>

typedef struct Technicals
{
    float step_size;
    float min_distance;
    float EPS;
};


typedef struct Params
{
    // Hybrid1
    float p[17];// = { 0.f,0.f,0.f,1.f,1.f,1.f,3.475f,2.f,1.f,0.f,1.65f,1.2f,1.f,0.5f,1.f,10.f,-4.f };
    // OctKoch (slow)
    //float p[17] = { 0.f,0.f, 0.f, 0.f, 2.4f, -0.75f, 0.25f, 0.25f, 1.f, 2.f, 2.f, 1.f, 1.f,0.f,0.f,0.f,0.f };
    // Menger Smooth
    //float p[17] = { 3.f, 1.f, 1.f, 1.f, 0.5f, 0.005f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,0.f };
};


class Camera
{
public:
    float3 position; // ray origin
    float3 lookat;
    Technicals tech;
    Params params;
    cudaGraphicsResource* cudapbo;
    void (*render_options)(Camera&);
    void (*reset_parameters)(Camera&);
    unsigned int num_pictures;
private:
    float3 right;
    float3 up;
    float3 mov_direction;
    float MOV_AMT = 0.1f; // how slowly we are moving (zooming)
    float ROT_AMT = 0.02f;
    int is_moving = 0; // only needed bc it feels better when you can freely press 2 buttons and still move
    enum MOVE_TYPE { MOVE_FORWARD, MOVE_BACKWARD, MOVE_UP, MOVE_DOWN, MOVE_RIGHT, MOVE_LEFT, ROTATE_RIGHT, ROTATE_LEFT, };
    MOVE_TYPE move_type;
public:
    Camera(cudaGraphicsResource* cudapbo, float3 position, float3 lookat) 
        : cudapbo(cudapbo), position(position), lookat(lookat)
    {
        this->tech.EPS = 0.004f;
        this->tech.step_size = 1.f;
        this->tech.min_distance = 0.001f;
        this->up = make_float3(0.f, 1.f, 0.f);
        this->right = make_float3(1.f, 0.f, 0.f);
        this->mov_direction = make_float3(0.f, 0.f, 0.f);
    }
    void KernelLauncher();
    void save_params();
    void load_params();
    void save_init_file();
    void load_init_file();
    void move();
    GLFWwindow* Init(GLuint* pbo_ptr);
    void Cleanup(GLuint* pbo_ptr);
private:
    void keyboardPressed(GLFWwindow* window, int key, int scancode, int action, int mods);
};


