#include "Camera.h"

#include "vendor/stb_image_write.h"

// for writing parameter files 
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


GLFWwindow* Camera::Init(GLuint* pbo_ptr)
{
    cudaGraphicsResource** cudapbo_ptr = &cudapbo;
    // standard OpenGL
    const char* glsl_version = "#version 430";
    if (!glfwInit()) exit(EXIT_FAILURE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "C U D A", NULL, NULL);
    if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // VSYNC
    if (glewInit() != GLEW_OK) exit(EXIT_FAILURE);
    std::cout << glGetString(GL_VERSION) << std::endl;


    // Cuda + GL interop setup
    glGenBuffers(1, pbo_ptr); // make & register PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo_ptr);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte) * WIDTH * HEIGHT, NULL, GL_DYNAMIC_DRAW);// unsigned byte is what we end up displaying.
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&(*cudapbo_ptr), *pbo_ptr, cudaGraphicsRegisterFlagsWriteDiscard));// this flag is efficient


    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);


    // funky glfw + class mix:
    glfwSetWindowUserPointer(window, this);
    auto func = [](GLFWwindow* w, int key, int scancode, int action, int mods)
    {
        static_cast<Camera*>(glfwGetWindowUserPointer(w))->keyboardPressed(w, key, scancode, action, mods);
    };
    glfwSetKeyCallback(window, func);
    KernelLauncher(); // render first frame
    return window;
}


void Camera::keyboardPressed(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {

        switch (key)
        {
        case GLFW_KEY_F: // F for F O T O G R A P H 
        {
            size_t fb_size = WIDTH * (HEIGHT - HEIGHT_OFFSET) * sizeof(uchar4);
            uchar4* h_fb = (uchar4*)malloc(fb_size);
            uchar4* dev_map;
            gpuErrchk(cudaGraphicsMapResources(1, &cudapbo, NULL));
            gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));
            cudaMemcpy(h_fb, dev_map, fb_size, cudaMemcpyDeviceToHost);
            gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));

            std::string file = "C:\\Users\\pwnag\\Desktop\\test";
            file.append(std::to_string(num_pictures));
            file.append(".bmp");
            int SUCCESS = stbi_write_bmp(file.c_str(), WIDTH, (HEIGHT - HEIGHT_OFFSET), 4, h_fb);
            free(h_fb);
            num_pictures++;
            save_init_file();
            break;
        }
            //   T E C H N I C A L   P A R A M E T E R S 
        case GLFW_KEY_V: // inc min distance 
            tech.min_distance *= 2.f; break;
        case GLFW_KEY_B: // dec min distance 
            tech.min_distance /= 2.f; break;
        case GLFW_KEY_N: // inc EPSILON
            tech.EPS *= 1.5f; break;
        case GLFW_KEY_M: // dec EPSILON
            tech.EPS /= 1.5f; break;
        case GLFW_KEY_K: // inc step size
            tech.step_size /= 0.9f; break;
        case GLFW_KEY_L: // dec step size
            tech.step_size *= 0.9f; break;
            //   N O R M A L   F U N C T I O N A L I T Y 
        case GLFW_KEY_Z: // save parameters
            save_params(); break;
        case GLFW_KEY_X: // load most recent parameter
            load_params(); break;
        case GLFW_KEY_R: // reset the params to original values 
            tech.step_size = 1.0f;
            tech.EPS = 0.004f;
            tech.min_distance = 0.001f; break;
        case GLFW_KEY_O: // move slower (zoom in)
            MOV_AMT *= 0.5f; break;
        case GLFW_KEY_P: // move faster (zoom out)
            MOV_AMT /= 0.5f; break;

        case GLFW_KEY_RIGHT: move_type = ROTATE_RIGHT; is_moving++; break;
        case GLFW_KEY_LEFT: move_type = ROTATE_LEFT; is_moving++; break;
        case GLFW_KEY_W: move_type = MOVE_FORWARD; is_moving++; break;
        case GLFW_KEY_S: move_type = MOVE_BACKWARD; is_moving++; break;
        case GLFW_KEY_D: move_type = MOVE_RIGHT; is_moving++; break;
        case GLFW_KEY_A: move_type = MOVE_LEFT; is_moving++; break;
        case GLFW_KEY_E: move_type = MOVE_UP; is_moving++; break;
        case GLFW_KEY_Q: move_type = MOVE_DOWN; is_moving++; break;
        case GLFW_KEY_ESCAPE: exit(420); break;
        default:break;
        }
        KernelLauncher();
    }
    if (action == GLFW_RELEASE)
    {
        switch (key)
        {
        case GLFW_KEY_RIGHT:
        case GLFW_KEY_LEFT:
        case GLFW_KEY_W:
        case GLFW_KEY_S:
        case GLFW_KEY_D:
        case GLFW_KEY_A:
        case GLFW_KEY_E:
        case GLFW_KEY_Q: is_moving--; break;
        default:break;;
        }
        KernelLauncher();
    }
}


void Camera::Cleanup(GLuint* pbo_ptr)
{
    cudaGraphicsResource** cudapbo_ptr = &cudapbo;
    // for some ungodly reason, putting this line in a function gives linking errors.
    //glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // unbind the currently bound buffer (pbo)
    gpuErrchk(cudaGLUnregisterBufferObject(*pbo_ptr));
    gpuErrchk(cudaGraphicsUnregisterResource(*cudapbo_ptr));
    glDeleteBuffers(1, pbo_ptr);
    glfwTerminate();
}


void Camera::move()
{
    if (is_moving)
    {
        switch (move_type) // this effectively lets you either move or turn at once. should i do both?
        {
        case MOVE_FORWARD:
            mov_direction = lookat - position;
            position += MOV_AMT * mov_direction;
            lookat += MOV_AMT * mov_direction;
            break;
        case MOVE_BACKWARD:
            mov_direction = position - lookat;
            position += MOV_AMT * mov_direction;
            lookat += MOV_AMT * mov_direction;
            break;
        case ROTATE_RIGHT:
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
        case ROTATE_LEFT: // see above 
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
        case MOVE_UP:
            position += MOV_AMT * up;
            lookat += MOV_AMT * up;
            break;
        case MOVE_DOWN:
            position -= MOV_AMT * up;
            lookat -= MOV_AMT * up;
            break;
        case MOVE_RIGHT:
            float3 right = cross(lookat - position, up); // make the right-vector
            lookat += MOV_AMT * right; // add right-vector to pos & lookat
            position += MOV_AMT * right;
            break;
        case MOVE_LEFT:
            float3 left = cross(up, lookat - position);
            lookat += MOV_AMT * left;
            position += MOV_AMT * left;
            break;
        }
#ifndef RENDER_EVERY_FRAME
        KernelLauncher();
    }
#else
}
    KernelLauncher();
#endif 
}



void Camera::save_params()
{
    // float to string 
    std::ostringstream ss;
    ss << position.x << " " << position.y << " " << position.z << " "
        << lookat.x << " " << lookat.y << " " << lookat.z << " "
        << MOV_AMT << " " << tech.step_size << " " << tech.min_distance << " " << tech.EPS << " "
        << params.p[0] << " " << params.p[1] << " " << params.p[2] << " " << params.p[3] << " " << params.p[4] << " " 
        << params.p[5] << " " << params.p[6] << " " << params.p[7] << " " << params.p[8] << " " << params.p[9] << " " 
        << params.p[10] << " " << params.p[11] << " " << params.p[12] << " " << params.p[13] << " " << params.p[14] << " " 
        << params.p[15] << " " << params.p[16] << "\n";
    std::string s(ss.str());

    // write to file 
    std::ofstream myfile;
    myfile.open(FILENAME);
    myfile << s.c_str();
    myfile.close();
    PRINT("saved params.");
}


void Camera::load_params()
{
    std::ifstream infile(FILENAME);
    float y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16;
    if (!(infile >> y1 >> y2 >> y3 >> y4 >> y5 >> y6 >> y7 >> y8 >> y9 >> y10 >> p0 >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9 >> p10 >> p11 >> p12 >> p13 >> p14 >> p15 >> p16))
    {
        PRINT("reading parameters F A I L E D");
    }
    position = make_float3(y1, y2, y3);
    lookat = make_float3(y4, y5, y6);
    MOV_AMT = y7;
    tech.step_size = y8;
    tech.min_distance = y9;
    tech.EPS = y10;
    params.p[0] = p0;
    params.p[1] = p1;
    params.p[2] = p2;
    params.p[3] = p3;
    params.p[4] = p4;
    params.p[5] = p5;
    params.p[6] = p6;
    params.p[7] = p7;
    params.p[8] = p8;
    params.p[9] = p9;
    params.p[10] = p10;
    params.p[11] = p11;
    params.p[12] = p12;
    params.p[13] = p13;
    params.p[14] = p14;
    params.p[15] = p15;
    params.p[16] = p16;
    PRINT("loaded params.");
}




void Camera::save_init_file()
{
    // float to string 
    std::ostringstream ss;
    ss << num_pictures << "\n";
    std::string s(ss.str());

    // write to file 
    std::ofstream myfile;
    myfile.open(INITFILENAME);
    myfile << s.c_str();
    myfile.close();
    PRINT("saved params.");
}


void Camera::load_init_file()
{
    // right now the init file is an int that says what picture # we are at. 
    std::ifstream infile(INITFILENAME);
    int i;
    if (!(infile >> i))
    {
        PRINT("reading init file F A I L E D");
    }
    num_pictures = i;
    PRINT("loaded init file.");
}



