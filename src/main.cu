

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "definitions.h"
#include "camera.h"

#define BUFFER_DATA(i) ((char *)0 + i)

class Shader
{
public:
    unsigned int ID;
    // constructor generates the shader on the fly
    // ------------------------------------------------------------------------
    Shader(const char* vertexPath, const char* fragmentPath)
    {
        // 1. retrieve the vertex/fragment source code from filePath
        std::string vertexCode;
        std::string fragmentCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;
        // ensure ifstream objects can throw exceptions:
        vShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        try 
        {
            // open files
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;
            // read file's buffer contents into streams
            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();
            // close file handlers
            vShaderFile.close();
            fShaderFile.close();
            // convert stream into string
            vertexCode   = vShaderStream.str();
            fragmentCode = fShaderStream.str();
        }
        catch (std::ifstream::failure& e)
        {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        }
        const char* vShaderCode = vertexCode.c_str();
        const char * fShaderCode = fragmentCode.c_str();
        // 2. compile shaders
        unsigned int vertex, fragment;
        // vertex shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");
        // fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");
        // shader Program
        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");
        // delete the shaders as they're linked into our program now and no longer necessary
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }
    // activate the shader
    // ------------------------------------------------------------------------
    void use() 
    { 
        glUseProgram(ID); 
    }
    // utility uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value) const
    {         
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value); 
    }
    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value) const
    { 
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value); 
    }
    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value) const
    { 
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value); 
    }

private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type)
    {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM")
        {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else
        {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
};




// extern "C" void thread_spawner(unsigned char *dst, const int numSMs);
extern "C" void thread_spawner(uchar4 *dst, float3 ro, float3 lookat, Params params);



void reset_options(Camera& camera)
{
    camera.position = make_float3(0.f, 0.f, 6.f); 
    camera.lookat = make_float3(0.0f);

    camera.params.EPS = 0.001f;
    camera.params.step_size = 1.f;
    camera.params.min_distance = 0.001f;
    camera.params.p[0] = 1.9f; // fixed_radius2 = 1.9f;
    camera.params.p[1] = 0.1f; // min_radius2 = 0.1f;
    camera.params.p[2] =-2.8f; // scale = -2.8f;
    camera.params.p[3] = 1.0f; // folding_limit = 1.0f;
    camera.params.p[4] = 0.f;
    camera.params.p[5] = 0.f;
    camera.params.p[6] = 0.f;
    camera.params.p[7] = 0.f;
    camera.params.p[8] = 0.f;
    camera.params.p[9] = 0.f;
    camera.params.p[10]= 0.f;
    camera.params.p[11]= 0.f;
    camera.params.p[12]= 0.f;
    camera.params.p[13]= 0.f;
    camera.params.p[14]= 0.f;
    camera.params.p[15]= 0.f;
    camera.params.p[16]= 0.f;
}


void render_options(Camera& camera) 
{
    int interaction = 0; // becomes 1 if any slider was used, so we know to render

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    ImGui::Begin("V/B inc/dec min_dist, N/M inc/dec EPS, K/L inc/dec step_size");
    if (ImGui::Button("Reset"))
    {
        reset_options(camera);
        camera.launch_kernel();
    }
    interaction |= ImGui::SliderFloat("fixed_radius2", &camera.params.p[0], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("min_radius2", &camera.params.p[1], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("scale", &camera.params.p[2], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("folding_limit", &camera.params.p[3], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p4", &camera.params.p[4], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p5", &camera.params.p[5], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p6", &camera.params.p[6], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p7", &camera.params.p[7], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p8", &camera.params.p[8], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p9", &camera.params.p[9], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p10", &camera.params.p[10], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p11", &camera.params.p[11], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p12", &camera.params.p[12], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p13", &camera.params.p[13], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p14", &camera.params.p[14], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p15", &camera.params.p[15], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p16", &camera.params.p[16], -5.f, 5.f);
    ImGui::End();

    if (interaction) camera.launch_kernel();
}


// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // // from GLUT:
    // glViewport(0, 0, w, h);
    // glMatrixMode(GL_MODELVIEW);
    // glLoadIdentity();
    // glMatrixMode(GL_PROJECTION);
    // glLoadIdentity();
    // glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    // // Do not call when window is minimized that is when
    // // width && height == 0
    // if (w != 0 && h != 0) initOpenGLBuffers(w, h);
    // glutPostRedisplay();


    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// settings
const unsigned int SCR_WIDTH = WIDTH;
const unsigned int SCR_HEIGHT = HEIGHT * 1.5;

int main()
{
    // camera
    Camera camera;
    camera.render_options = render_options;
    camera.reset_options = reset_options;

    // GLFW
    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "soy", NULL, NULL);
    if (!window) { glfwTerminate(); return -1;}
    glfwMakeContextCurrent(window);
    glfwSetErrorCallback(glfw_error_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    // glfwSwapInterval(1); // Enable vsync
    // set keyboard callback to camera's function using a closure
    glfwSetWindowUserPointer(window, &camera);
    auto func = [](GLFWwindow* w, int key, int scancode, int action, int mods)
    {
        static_cast<Camera*>(glfwGetWindowUserPointer(w))->keypress_callback(key, action);
    };
    glfwSetKeyCallback(window, func);

    // IMGUI
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("../Common/vendor/learn_opengl/4.1.texture.vs", "../Common/vendor/learn_opengl/4.1.texture.fs"); 

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions         // texture coords
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {  
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);




    setenv("DISPLAY", ":0", 0);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
    int numSMs = deviceProp.multiProcessorCount; // number of multiprocessors
    std::cout << "num SMs:" << numSMs << std::endl;



    // create GL texture 
    GLuint gl_Tex, gl_PBO; // OpenGL PBO and texture "names"
    struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
    uchar4 *d_dst = NULL;  // Destination image on the GPU side
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    unsigned char* h_Src = (unsigned char*)malloc(SCR_WIDTH * SCR_HEIGHT * 4 * sizeof(unsigned char)); // Source image on the host side
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);

    // create PBO
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, h_Src, GL_STREAM_COPY);
    // While a PBO is registered to CUDA, it can't be used
    // as the destination for OpenGL drawing calls.
    // But in our particular case OpenGL is only used
    // to display the content of the PBO, specified by CUDA kernels,
    // so we need to register/unregister it only once.
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));


    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("OpenGL Renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL Vendor: %s\n", glGetString(GL_VENDOR));
    printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));



    ourShader.use();
    glBindVertexArray(VAO);





    // render first frame
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, 0, cuda_pbo_resource));
    camera.dst = d_dst;
    camera.cuda_pbo_resource = cuda_pbo_resource;

    // // prepare the drawing memory
    // dim3 threads(BLOCKDIM_X, BLOCKDIM_Y); // # 256 threads per block
    // dim3 grid((WIDTH / BLOCKDIM_X), ((HEIGHT - HEIGHT_OFFSET)/ BLOCKDIM_Y)); // 800 blocks in 1 grid
    // pre_kernel<<<grid, threads>>>(d_dst); 

    camera.launch_kernel();
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    camera.reset_options(camera); // initialize fractal params
    camera.launch_kernel(); // render first frame




    int display_w, display_h;
    // while (!glfwWindowShouldClose(window))
    while (camera.should_render)
    {
        // CUDA rendering
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
        // thread_spawner(d_dst, numSMs);
        camera.move();
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
        // load texture from PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
        glBindTexture(GL_TEXTURE_2D, gl_Tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


        // OpenGL triangles
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // Rendering
        camera.render_options(camera);
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        processInput(window);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    if (h_Src) {
        free(h_Src);
        h_Src = 0;
    }
    if (gl_Tex) {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }
    if (gl_PBO) {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}



