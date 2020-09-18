// Sources:
// CUDA + GLFW setup from: https://gist.github.com/kamino410/09df4ecdf37b03cbd05752a7b2e52d3a
// Mandelbox Ryu: https://www.shadertoy.com/view/XdlSD4
// Mandelbox Colored: https://www.shadertoy.com/view/3sj3RK
// glow / ambient occlusion: https://github.com/HackerPoet/PySpace/blob/master/pyspace/frag.glsl
// my favorite ambient occlusion: https://www.shadertoy.com/view/3sGXzD
// TgladFold & more formulas: https://sites.google.com/site/mandelbulber/user-manual
// OctFold formula from Mandelbulb3d and from the Mandelbulber source code, and lots of painful experimentation


// ===== TODO: 
// -check hybrids on fragmentaruim code - under knighty collection, with menger
//      - apparently they all use numerical grad, see syntoipia part 5[T R Y A G A I N]
// - implement trap functions from syntopia 8
// - implement my own hybrid from before (easy) 
// - read Syntopia blogs
// - check out that 1/exp(p) DE for mandelbox on Fractal Forums (see Sy ntopia blog)
// - try to understand hybrids in Mandelbulber / woofractal
// - MY SHADOW IS BROKEN BECAUSE SOFT SHADOW WORKS GREAT!!!!!
// NOTE: the version used before OctKoch is "_AmazingBox" 
// soft shadows decrease the FPS quite a bit, with AO not that necessary.
// GLOW: keep track of minimum that DE ever got (code parade) if ray never hit object, we know how close it got - glow can be based on that
// Doubles: I loaded a low-res area and it seemed to be improved? although hard to tell bc its so slow. Maybe for offline render...
// Reflections: cool but wonky. the minimal step required for the DE to not catch itself also ruins the zooming experience. otherwise, too noisy
// fixed nvprof, so i can use it when my stuff gets more complex :D
// well i just rediscovered Glow lol. wanted to debug and see why I get noise, so i output number of steps to Red. now it looks hard core. 
// i think a good combination of step size and EPS makes a good bulb. looks like the distance field is getting 
// warped in teh background - but I know that high loop iteration fixes that... so there really is no problem i guess?
// either use an IMGUI slider to cut the ugly background, or install a floor like BoardIFS to look at.
// ===== TODO L8R:
// - global illumination
// - meta programming: given a list of params and a DE, create a cuda kernel for that fractal
//      the meta program also includes a program that saves & reads the params.. probably should be a class...
// - try to understand DIFS 
// - re-download the difs explanations by darkbeam
// - chromatic aberration 
// - depth of feel 
// - color the fractal (orbit traps)
// - render images (make list of params -> interpolate -> render list -> render animations)
// - check if GL_DYNAMIC_DRAW or variant is faster once i have a FPS counter / nvprof
// - use quaternions for camera rotation 
// puzzle - how to zoom to pts near 1 with precision of float's 0?
// puzzle - why does dust appear (max out number of steps) in OctKoch? 
//      my hypothesis is based on that picture from AOC - ray marcher takes very many steps for rays close to parallel to a surface
//      and im guessing there must be lots of holes here where rays get trapped in a "tunnel" and all take 128 steps?
//      the REAL puzzle is, how can Mandelbulb3D not have these bugs??? :O
//      this has introduced a lot of bugs and ideas, perhaps Monte Carlo, plus replacing step counter with while loop.


#include "Common.cuh"
#include "Common.cu" // yep
#include "Camera.h"

#ifdef INTERACTIVE
#include "Fractals/Renderer_GUI.cuh"
#else 
#include "Fractals/Renderer_Offline.cuh" 
#endif

#include "Fractals/Parameters.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"



#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
        exit(99);
    }
}




int main(void)
{
    cudaGraphicsResource* cudapbo;
    Camera camera = Camera(cudapbo, make_float3(0.f, 0.f, -7.f), make_float3(0.f, 0.f, -6.f));// camera and window are intertwined

    // set which fractal to display
    camera.render_options = render_options_Hybrid1;
    camera.reset_parameters = reset_parameters_Hybrid1;
    camera.reset_parameters(camera);

#ifdef INTERACTIVE

    GLuint pbo;
    GLFWwindow* window = camera.Init(&pbo);

    while (!glfwWindowShouldClose(window))
    {
        camera.move();
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        camera.render_options(camera);

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);//(see note in camera.Cleanup())
    camera.Cleanup(&pbo);
    return 0;
#else 
    // render with STB_Image 
    camera.load_params(); // load scene 
    dim3 threads(8, 8);
    dim3 grids(WIDTH / 8, HEIGHT / 8); // 96 x 64
    size_t fb_size = WIDTH * HEIGHT * sizeof(uchar4);
    /*
    camera.tech.min_distance *= .8f;
    camera.tech.EPS *= .8f;
    camera.tech.step_size *= 0.9f * 0.9f;
    */
    uchar4* h_fb = (uchar4*)malloc(fb_size);
    uchar4* d_fb = 0;
    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));
    checkCudaErrors(cudaMemcpy(d_fb, h_fb, fb_size, cudaMemcpyHostToDevice));
    kernel << <grids, threads >> > (d_fb, 0, camera.position, camera.lookat, camera.tech, camera.params);
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
    int SUCCESS = stbi_write_bmp("C:\\Users\\pwnag\\Desktop\\test.bmp", WIDTH, HEIGHT, 4, h_fb);
    checkCudaErrors(cudaFree(d_fb));
    free(h_fb);
    return SUCCESS;
#endif
}