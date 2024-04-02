#include "camera.h"
#include "helper_math.h"
#include <GLFW/glfw3.h> 

#include <helper_cuda.h>

// #include "file_ops.h"

// save image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


extern "C" void thread_spawner(uchar4*, float3, float3, Params);


// take photograph
void Camera::take_photograph()
{
    // get file size
    size_t fb_size = WIDTH * HEIGHT * sizeof(uchar4);
    uchar4* h_fb = (uchar4*)malloc(fb_size);
    uchar4* dev_map;
    
    // move from GPU to CPU
    if (this->is_mapped_pbo == false)
    {
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        this->is_mapped_pbo = true;
    }
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, 0, cuda_pbo_resource));
    cudaMemcpy(h_fb, dev_map, fb_size, cudaMemcpyDeviceToHost);
    if (this->is_mapped_pbo == true)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        this->is_mapped_pbo = false;
    }
    

    // save the buffer to an image file
    std::string file = "frames/fractal_img_";
    file.append(std::to_string(frame_number));
    file.append(".bmp");
    stbi_flip_vertically_on_write(1);
    int SUCCESS = stbi_write_bmp(file.c_str(), WIDTH, HEIGHT, 4, h_fb);
    free(h_fb);

    this->frame_number += 1;
}


Camera::Camera()
{
    //position = make_float3(0.f, 0.f, 6.f); 
    position = make_float3(-3.75598f, 1.7928e-08f, -4.69161f); 
    //lookat = make_float3(0.0f);
    lookat = make_float3(-3.51827f, 1.7928e-08f, -3.72027);
    // fm = file_manager();
}


void Camera::launch_kernel()
{
    thread_spawner(dst, position, lookat, params);
}


void Camera::keypress_callback(int key, int action)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        //   T E C H N I C A L   P A R A M E T E R S 
        // inc/dec min distance 
        case GLFW_KEY_V: params.min_distance *= 2.f; break;
        case GLFW_KEY_B: params.min_distance /= 2.f; break;
        // inc/dec EPSILON
        case GLFW_KEY_N: params.EPS *= 1.5f; break;
        case GLFW_KEY_M: params.EPS /= 1.5f; break;
        // inc/dec step size
        case GLFW_KEY_K: params.step_size /= 0.9f; break;
        case GLFW_KEY_L: params.step_size *= 0.9f; break;
        // reset technical params
        case GLFW_KEY_R: 
            params.EPS = 0.001f;
            params.step_size = 1.0f;
            params.min_distance = 0.001f; break;

        //   N O R M A L   F U N C T I O N A L I T Y 
        // save/restore parameters parameters
        case GLFW_KEY_Z:
        {
            // // get the number of the last saved params file 
            // int last_saved = fm.read_int_from_init_file();
            // std::string s("fractal_params_");
            // s += std::to_string(last_saved + 1); // increment, next file
            // s += std::string(".bin");

            // // write data to this file
            // FILE *my_file = fopen(s.c_str(), "wb");
            // if (my_file) 
            // {
            //     fwrite(&position, sizeof(float3), 1, my_file);
            //     fwrite(&lookat,   sizeof(float3), 1, my_file);
            //     fwrite(&params,   sizeof(Params), 1, my_file);
            //     fclose(my_file);
            // }
            // else printf("ERROR OPENING FILE TO WRITE");

            // take a photograph
            take_photograph();

            // // update the init file saying we created 1 additional file
            // fm.write_int_to_init_file(last_saved + 1);
            break;
        }
        case GLFW_KEY_X:
        {
            // // get the number of the last saved params file 
            // int last_saved = fm.read_int_from_init_file();
            // std::string s("fractal_params_");
            // s += std::to_string(last_saved); 
            // s += std::string(".bin");

            // // read this data 
            // FILE *my_file = fopen(s.c_str(), "rb");
            // if (my_file)
            // {
            //     fread(&position, sizeof(float3), 1, my_file);
            //     fread(&lookat,   sizeof(float3), 1, my_file);
            //     fread(&params,   sizeof(Params), 1, my_file);
            //     fclose(my_file);
            //     launch_kernel(); // render with the restored params
            // }
            // else printf("ERROR OPENING FILE TO READ");
            break;
        }
        // move slower/faster (zoom in/out)
        case GLFW_KEY_O: MOV_AMT *= 0.5f; break;
        case GLFW_KEY_P: MOV_AMT *= 2.0f; break;
        // movement in the 3D space
        case GLFW_KEY_RIGHT: move_type = ROTR; is_moving++; break;
        case GLFW_KEY_LEFT:  move_type = ROTL; is_moving++; break;
        case GLFW_KEY_W:     move_type = MOVF; is_moving++; break;
        case GLFW_KEY_S:     move_type = MOVB; is_moving++; break;
        case GLFW_KEY_D:     move_type = MOVR; is_moving++; break;
        case GLFW_KEY_A:     move_type = MOVL; is_moving++; break;
        case GLFW_KEY_E:     move_type = MOVU; is_moving++; break;
        case GLFW_KEY_Q:     move_type = MOVD; is_moving++; break;
        // quit 
        case GLFW_KEY_ESCAPE: this->should_render = false; break;
        }
    }
    if (action == GLFW_RELEASE)
    {
        switch (key)
        {
        // when we release a key, that means we stop moving
        case GLFW_KEY_RIGHT:
        case GLFW_KEY_LEFT:
        case GLFW_KEY_W:
        case GLFW_KEY_S:
        case GLFW_KEY_D:
        case GLFW_KEY_A:
        case GLFW_KEY_E:
        case GLFW_KEY_Q: is_moving--; break;
        }
    }
}


void Camera::move()
{
    if (is_moving)
    {
        switch (move_type)
        {
        case MOVF:
        {
            float3 mov_direction = lookat - position;
            position += MOV_AMT * mov_direction;
            lookat += MOV_AMT * mov_direction;
            break;
        }
        case MOVB:
        {
            float3 mov_direction = position - lookat;
            position += MOV_AMT * mov_direction;
            lookat += MOV_AMT * mov_direction;
            break;
        }
        case ROTR:
        {
            float theta = ROT_AMT; // TODO: this is stuck in XZ plane, need to make it relative to current orientation
            // move position to origin 
            float x1 = lookat.x - position.x;
            float z1 = lookat.z - position.z;
            // rotate 
            float x2 = cos(theta) * x1 - sin(theta) * z1;
            float z2 = sin(theta) * x1 + cos(theta) * z1;
            // move position back 
            lookat.x = position.x + x2;
            lookat.z = position.z + z2;
            break;
        }
        case ROTL: // see above 
        {
            float theta = -ROT_AMT;
            float x1 = lookat.x - position.x;
            float z1 = lookat.z - position.z;
            float x2 = cos(theta) * x1 - sin(theta) * z1;
            float z2 = sin(theta) * x1 + cos(theta) * z1;
            lookat.x = position.x + x2;
            lookat.z = position.z + z2;
            break;
        }
        case MOVU:
        {
            const float3 up = make_float3(0.f, 1.f, 0.f);
            position += MOV_AMT * up;
            lookat += MOV_AMT * up;
            break;
        }
        case MOVD:
        {
            const float3 up = make_float3(0.f, 1.f, 0.f);
            position -= MOV_AMT * up;
            lookat -= MOV_AMT * up;
            break;
        }
        case MOVR:
        {
            const float3 up = make_float3(0.f, 1.f, 0.f);
            float3 right = cross(lookat - position, up); // make the right-vector
            lookat += MOV_AMT * right; // add right-vector to pos & lookat
            position += MOV_AMT * right;
            break;
        }
        case MOVL:
        {
            const float3 up = make_float3(0.f, 1.f, 0.f);
            float3 left = cross(up, lookat - position);
            lookat += MOV_AMT * left;
            position += MOV_AMT * left;
            break;
        }
        }

        launch_kernel();
    }
}


