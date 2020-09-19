# Fractal-Explorer
a real-time fractal explorer made with CUDA!
this project is on hiatus because it seems that working on this project has broken one of my laptop fans. 

![](https://github.com/ConsciousMachines/Fractal-Explorer/blob/master/Screenshot%20(280).png)

![](https://github.com/ConsciousMachines/Fractal-Explorer/blob/master/Screenshot%20(278).png)

![](https://github.com/ConsciousMachines/Fractal-Explorer/blob/master/Screenshot%20(284).png)

# Usage
- WASD -> movement in the XZ plane
- Left/Right arrow keys -> rotate left/right
- Q/E -> move up/down along the Y axis 
- O/P -> slow down / speed up movement. This has the effect of zooming in/out, as you slow down while approaching a structure. 
- F -> take a photograph of the current screen and save as .bmp
- Z -> save the fractal parameters, and your location (basically you can save whatever you are seeing). But there is only one save file!
- X -> load save file. 
- K/L, N/M, V/B -> technical rendering stuff: step_size, epsilon for determining if you're near enough to the fractal, and another epsilon used in finding normals. These can change the appearance of the object, smaller epsilons will show finer structure but can generate noise. step_size wasn't particularly useful.
- R -> reset the technical parameters 


# How to build
- link with GLEW, GLFW, CUDA runtime. 
- define "GLEW_STATIC" in preprocessor directives
- also requires STB_Image and ImGui
