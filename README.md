# Fractal Explorer

This project aims to be a `rapid prototyping`, `real time` version of the preview panel of *Mandelbulb3D*, allowing the user to rapidly explore fractal structures by instantly changing parameters using the ImGui sliders, and move the camera using the arrow keys to find aesthetic angles.

![](https://github.com/ConsciousMachines/Fractal-Explorer/blob/master/img/fractal_animation_large.gif)

Rather than waiting to render an animation, you can fly around the structure in real time. This allows you to find coordinates and parameters that you can then import into Mandelbulb3D for professional lighting. 

![](https://github.com/ConsciousMachines/Fractal-Explorer/blob/master/img/img_1.png)

![](https://github.com/ConsciousMachines/Fractal-Explorer/blob/master/img/img_2.png)

![](https://github.com/ConsciousMachines/Fractal-Explorer/blob/master/img/img_3.png)

# Distance Estimators 

The part of the code that determines the fractal is the DE function. It takes a pixel's position `z`, and a struct `params` which holds parameters of the fractal. These are linked to the ImGui sliders so you can change them in real time. To change the fractal, you replace this function.

```c++
__device__ float DE(float3 z, Params params) {

    float fixed_radius2 = params.p[0];
    float min_radius2 = params.p[1];
    float scale = params.p[2];
    float folding_limit = params.p[3];

    float3 offset = z;
    float dr = 1.0f;
    for(int n = 0; n < 15; ++n) {
        // box fold 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;

        // sphere fold
        float r2 = dot(z, z);
        if(r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
        }else if(r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
        }

        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0f;
    }
    float r = length(z);
    return r / abs(dr);
}
```

# Lighting 

Since the point of this project is rapid prototyping, we use a simple yet aesthetic lighting scheme found on `ShaderToy`. 

# Usage
- `WASD`: movement in the XZ plane
- `Left/Right arrow keys`: rotate left/right
- `Q/E`: move up/down along the Y axis 
- `O/P`: slow down / speed up movement. This has the effect of zooming in/out, as you slow down while approaching a structure. 
- `F`: take a photograph of the current screen and save as `.bmp` in the `frames` folder. 
- `Z`: save the fractal parameters, and your location (basically you can save whatever you are seeing). But there is only one save file!
- `X`: load save file. 
- `K/L, N/M, V/B`: technical rendering stuff: `step_size`, `epsilon` for determining if you're near enough to the fractal, and another `epsilon` used in finding normals. These can change the appearance of the object, smaller `epsilon`s will show finer structure but can generate noise. `step_size` wasn't particularly useful.
- `R`: reset the technical parameters 

# How to build
- `Dependencies`: GLAD, GLFW, CUDA, STB_Image, ImGui.
