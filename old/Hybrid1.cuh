#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// ------------------------------------------------ M A N D E L B O X   R Y U ------------------
// ---------------------------------------------------------------------------------------------


__device__ inline float DE(float3 z)
{
    //z.xz = rotate(z.xz, 0.1 * iTime);

    // menger params
    float3 MengerOffset = make_float3(1.0f);
    float MengerScale = 3.475f;//3.0;

    // mandelbox params
    float fixed_radius2 = 1.f;
    float min_radius2 = 0.f;
    float folding_limit = 1.65f;
    float MandelboxScale = 1.2f;// 1.2f + 0.2f * sin(0.5 * iTime);
    float3 MandelboxOffset = z;
    float dr = 1.0f;
    float r2;

    // shared
    int Iterations = 4;
    int n = 0;
    float both_scale = 1.0f;
    while (n < Iterations)
    {
        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs


        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs


        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs


        // mandelbox step 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
            both_scale *= temp;
        }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        //MandelboxScale = -2.8 - 0.2 * iTime;
        both_scale *= MandelboxScale; // maybe use abs






        // menger step
        z = abs(z);
        if (z.x < z.y) {
            float temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            float temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            float temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - 1.0f); // same space transform as tetrahedron
        both_scale *= MengerScale;
        //dr *= MengerScale;

        if (z.z < -0.5f * MengerOffset.z * (MengerScale - 1.0f))
        {
            z.z += MengerOffset.z * (MengerScale - 1.0f);
        }

        // menger step
        z = abs(z);
        if (z.x < z.y) {
            float temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) { 
            float temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) { 
            float temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - 1.0f); // same space transform as tetrahedron
        both_scale *= MengerScale;
        //dr *= MengerScale;

        if (z.z < -0.5f * MengerOffset.z * (MengerScale - 1.0f))
        {
            z.z += MengerOffset.z * (MengerScale - 1.0f);
        }


        n++;
    }
    return length(z) / abs(both_scale + 1.f);

}



























__device__ inline float DE_Mandelbox(float3 z) {
    float fixed_radius2 = 1.9f;
    float min_radius2 = 0.1f;
    float scale = -2.8f;
    float3 offset = z;
    float dr = 1.0f;
    for (int n = 0; n < 15; ++n) {
        z = clamp(z, -1.f, 1.f) * 2.0f - z;
        
        float r2 = dot(z, z);
        if (r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
        }
        else if (r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
        }
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0f;
    }
    return length(z) / abs(dr);
}


__device__ inline float DE_Menger(float3 z) // Menger from Syntopia
{
    float Scale = 3.f;
    float3 Offset = make_float3(1.f);// 0.92858f, 0.92858f, 0.32858f);
    float d = 1000.0;
    for (int n = 0; n < 15; n++) {
        z = abs(z);
        if (z.x < z.y) {
            float temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            float temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            float temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = Scale * z - Offset * (Scale - 1.0f);
        if (z.z < -0.5f * Offset.z * (Scale - 1.0f))  z.z += Offset.z * (Scale - 1.0f);
        d = min(d, length(z) * pow(Scale, float(-n) - 1.0f));
    }

    return d - 0.001f;
}

__device__ inline float DE_failed_hybrid(float3 z)
{
    // menger params
    float MengerScale = 3.f;
    float3 MengerOffset = make_float3(1.f);// 0.92858f, 0.92858f, 0.32858f);
    float d = 1000.0;

    // mandelbox params
    float fixed_radius2 = 1.9f;
    float min_radius2 = 0.1f;
    float MandelboxScale = 3.f;// 2.8f;
    float3 MandelboxOffset = z;
    float dr = 1.0f;

    for (int i = 0; i < 5; i++)
    {
        // 4 mb
        z = clamp(z, -1.f, 1.f) * 2.0f - z;
        float r2 = dot(z, z);
        if (r2 < min_radius2) { float temp = (fixed_radius2 / min_radius2); z *= temp; dr *= temp; }
        else if (r2<fixed_radius2){float temp=(fixed_radius2 / r2);z *= temp;dr *= temp;}
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        z = clamp(z, -1.f, 1.f) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) { float temp = (fixed_radius2 / min_radius2); z *= temp; dr *= temp; }
        else if (r2 < fixed_radius2) { float temp = (fixed_radius2 / r2); z *= temp; dr *= temp; }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        z = clamp(z, -1.f, 1.f) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) { float temp = (fixed_radius2 / min_radius2); z *= temp; dr *= temp; }
        else if (r2 < fixed_radius2) { float temp = (fixed_radius2 / r2); z *= temp; dr *= temp; }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        z = clamp(z, -1.f, 1.f) * 2.0f - z;
        r2 = dot(z, z);
        if (r2 < min_radius2) { float temp = (fixed_radius2 / min_radius2); z *= temp; dr *= temp; }
        else if (r2 < fixed_radius2) { float temp = (fixed_radius2 / r2); z *= temp; dr *= temp; }
        z = MandelboxScale * z + MandelboxOffset;
        dr = dr * abs(MandelboxScale) + 1.0f;
        // 2 menger
        z = abs(z);
        if (z.x < z.y) {
            float temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            float temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            float temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - 1.0f);
        if (z.z < -0.5f * MengerOffset.z * (MengerScale - 1.0f))  z.z += MengerOffset.z * (MengerScale - 1.0f);
        //d = min(d, length(z) * pow(MengerScale, float(-n) - 1.0f));
        dr *= MengerScale;

        z = abs(z);
        if (z.x < z.y) {
            float temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            float temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            float temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - 1.0f);
        if (z.z < -0.5f * MengerOffset.z * (MengerScale - 1.0f))  z.z += MengerOffset.z * (MengerScale - 1.0f);
        //d = min(d, length(z) * pow(MengerScale, float(-n) - 1.0f));
        dr *= MengerScale;

    }
    return length(z) / abs(dr);

}


__device__ inline float intersect(float3 ro, float3 rd, float step_size)
{
    float res;
    float t = 0.f;
    for (int i = 0; i < 120; i++)
    {
        float3 p = ro + rd * t;
        res = DE(p);

        if (res < 0.001f * t || res > 20.f) break;
        t += res*step_size;
    }

    if (res > 20.f) t = -1.f;
    return t;
    //return i;
}


__device__ inline float3 lighting(float3 p, float3 rd, float ps, float3 light_dir) {

    //float3 l1_dir = normalize(make_float3(0.8f, 0.8f, -1.4f));
    //float3 l1_dir = normalize(make_float3(0.f, 0.f, -5.f));
    float3 l1_dir = light_dir;
    float3 l1_col = 0.3f * make_float3(1.5f, 1.69f, 0.79f);

#ifdef BONUS
    float3 l2_dir = normalize(make_float3(-0.8f, 0.5f, 0.3f));
    float3 l2_col = make_float3(0.89f, 0.99f, 1.3f);
#endif
    float3 e0 = make_float3(ps, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, ps, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, ps);
    float3 n = normalize(make_float3(
        DE(p + e0) - DE(p - e0),
        DE(p + e1) - DE(p - e1),
        DE(p + e2) - DE(p - e2)));

    float shadow = 1.f;//softshadow(p, l1_dir, 10.0 );

    float dif1 = max(0.0f, dot(n, l1_dir));
#ifdef BONUS
    float dif2 = max(0.0f, dot(n, l2_dir));
    float bac1 = max(0.3f + 0.7f * dot(make_float3(-l1_dir.x, -1.0f, -l1_dir.z), n), 0.0f);
    float bac2 = max(0.2f + 0.8f * dot(make_float3(-l2_dir.x, -1.0f, -l2_dir.z), n), 0.0f);
    float spe = max(0.0f, pow(clamp(dot(l1_dir, reflect(rd, n)), 0.0f, 1.0f), 10.0f));
#endif
    float3 col = 5.5f * l1_col * dif1 * shadow;
#ifdef BONUS
    col += 1.1f * l2_col * dif2;
    col += 0.3f * bac1 * l1_col;
    col += 0.3f * bac2 * l2_col;
    col += 1.0f * spe;
#endif

    //float t=mod(p.y+0.1*texture(iChannel0,p.xz).x-time*150.0, 5.0);
    //col = mix(col, make_float3(6.0, 6.0, 8.0),
    //          pow(smoothstep(0.0, .3, t) * smoothstep(0.6, .3, t), 15.0));;
    return col;
}


__global__ void kernel(uchar4* map, unsigned int iTime, float3 _position, float3 _lookat, float step_size, float EPS) {
    //  UV coords
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // ranges from 0 to 768
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // ranges from 0 to 512
    int idx = ix + iy * WIDTH;//blockDim.x * gridDim.x; 
    float qx = ((float)ix) / WIDTH; // ranges from 0 to 1
    float qy = ((float)iy) / HEIGHT; // ranges from 0 to 1
    float uvx = ((qx - 0.5f) * 2.0f) * (WIDTH / HEIGHT);  // range from -1 to 1
    float uvy = ((qy - 0.5f) * 2.0f); // range from -1 to 1


    // camera
    //float stime = sin(iTime * 0.1f); float ctime = cos(iTime * 0.1f); float time = iTime * 0.01f;
    float3 ro = _position;
    float3 lookat = _lookat;
    float3 f = normalize(lookat - ro); //   F O R W A R D 
    float3 r = cross(f, make_float3(0.0f, 1.0f, 0.0f)); //   S I D E ? TODO: left or right, right uses diff order in cross prod
    float3 u = cross(r, f); // UP ? TODO: also uses diff order in cross prod
    float3 rd = normalize(uvx * r + uvy * u + 2.8f * f);  // transform from view to world



    float3 mtl = make_float3(1.0f, 1.3f, 1.23f) * 0.8f; // green color
    float3 col = make_float3(1.0f, 1.0f, 1.0f);
    float t = intersect(ro, rd, step_size);
    if (t > -0.5f) {
        col = lighting(ro + t * rd, rd, EPS, _position - _lookat) * mtl * 0.2f; // light coming from camera :D
    }
    //t *= 5.f;
    //unsigned char result = (unsigned char)min(t, 250.f);

    col = clamp(col, 0.f, 1.f); // sev genius input. there is a bit of overflow somewhere but this fixes it.
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