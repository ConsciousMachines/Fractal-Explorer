#include "../Common.cuh"
#include "../Camera.h"

#include <curand.h>
#include <curand_kernel.h>


// -------------------------------------------- H Y B R I D   # 1 ------------------------------
// ---------------------------------------------------------------------------------------------

// COOL FRACTALS:
// Hybrid1 := loop 4 {4 mandelbox, 1 menger} - has bulbs
// Hybrid2 := loop 4 {1 menger, 4 mandelbox} - super clean, Manger Offset Y from hybrid preset


// TODO:
// - add repeat (mod) to Mandelbox Offset (pass it the argument iTime % 5)

// default is rainbow normals. instead can use AO or Glow.


//   C O L O R   O P T I O N S   -   p i c k   1 
//-------------------------------------------------------------
//#define AMBIENT_OCCLUSION
//#define RAINBOW
//#define RYU 
#define SEV // fails on Hybrid2, looks amazing on Hybrid1 ! ! ! 
//-------------------------------------------------------------

#define max_iter 12000.f


#define bone make_float3(0.89f, 0.855f, 0.788f)
#define BRIGHTNESS 1.2f
#define GAMMA 1.4f
#define SATURATION .65f




// Mandelbox with MengerSmt. needs experimentation, but has smooth ass surfaces!
__device__ inline float DE_(float3 z, Params params)
{
    // Mandelbox 
    float3 MandelboxOffset = make_float3(params.p[0], params.p[1], params.p[2]);
    float box_mult = params.p[7];
    float FixedR2 = params.p[8];
    float MinR2 = params.p[9];
    float fold = params.p[10];
    float MandelboxScale = params.p[11];
    float dr_offset_Mandel = params.p[14];
    float dr_offset_Menger = params.p[15];
    float dr_offset_final = params.p[16];

    // Menger Smt
    float sc = params.p[3];
    float sc1 = sc - params.p[4];
    float Cx = params.p[5];
    float Cy = params.p[6];
    float Cz = params.p[12];
    float s = params.p[13];
    float sc2 = sc1 / sc;
    float t = 9999999.0f;

    MandelboxOffset = z + MandelboxOffset;
    const int Iterations = 3;
    float dr = 1.f; // CORR
    float r2, temp; // local

    for (int n = 0; n < Iterations; n++)
    {

        for (int soy = 0; soy < 7; soy++)
        {
            // mandelbox step 
            z = clamp(z, -fold, fold) * box_mult - z;
            r2 = dot(z, z);
            if (r2 < MinR2) {
                temp = (FixedR2 / MinR2);
                z *= temp;
                dr *= temp;
            }
            else if (r2 < FixedR2) {
                temp = (FixedR2 / r2);
                z *= temp;
                dr *= temp;
            }
            z = MandelboxScale * z + MandelboxOffset;
            dr = dr * abs(MandelboxScale) + dr_offset_Mandel;
        }


        // MengerSmt step
        z.x = sqrt(z.x * z.x + s);
        z.y = sqrt(z.y * z.y + s);
        z.z = sqrt(z.z * z.z + s);
        t = z.x - z.y;
        t = 0.5f * (t - sqrt(t * t + s));
        z.x = z.x - t;
        z.y = z.y + t;
        t = z.x - z.z;
        t = 0.5f * (t - sqrt(t * t + s));
        z.x = z.x - t; z.z = z.z + t;
        t = z.y - z.z;
        t = 0.5f * (t - sqrt(t * t + s));
        z.y = z.y - t; z.z = z.z + t;
        z.z = z.z - Cz * sc2;
        z.z = -sqrt(z.z * z.z + s);
        z.z = z.z + Cz * sc2;
        z.x = sc * z.x - Cx * sc1;
        z.y = sc * z.y - Cy * sc1;
        z.z = sc * z.z;
        dr = dr * abs(sc);//+ 1.; // CORR

        // i removed an additional menger step bc the bulb is already here 
    }
    return length(z) / abs(dr + dr_offset_final);
}



__device__ inline float DE(float3 z, Params params)
{
    float3 MandelboxOffset = make_float3(params.p[0], params.p[1], params.p[2]);
    float3 MengerOffset = make_float3(params.p[3], params.p[4], params.p[5]);
    float MengerScale = params.p[6];
    float box_mult = params.p[7];
    float FixedR2 = params.p[8];
    float MinR2 = params.p[9];
    float fold = params.p[10];
    float MandelboxScale = params.p[11];
    float Menger_Scale_Offset = params.p[12];
    float Menger_Z_thing = params.p[13];
    float dr_offset_Mandel = params.p[14];
    //float dr_offset_Menger = params.p[15];
    //float dr_offset_final = params.p[16];
    float XZ_plane_pos = params.p[15];
    float XY_plane_pos = params.p[16];

    // NEW : XY - plane for cutting crap out 
    float dist_to_XZ_plane = XZ_plane_pos - z.y;
    float dist_to_XY_plane = z.z - XY_plane_pos;


    MandelboxOffset = z + MandelboxOffset;
    const int Iterations = 4;
    float dr = 1.0f;
    float r2, temp; // local

    for (int n = 0; n < Iterations; n++)
    {
        for (int soy = 0; soy < 4; soy++)
        {
            // mandelbox step 
            z = clamp(z, -fold, fold) * box_mult - z;
            r2 = dot(z, z);
            if (r2 < MinR2) {
                temp = (FixedR2 / MinR2);
                z *= temp;
                dr *= temp;
            }
            else if (r2 < FixedR2) {
                temp = (FixedR2 / r2);
                z *= temp;
                dr *= temp;
            }
            z = MandelboxScale * z + MandelboxOffset;
            dr = dr * abs(MandelboxScale) + dr_offset_Mandel;
        }

        // menger step
        z = abs(z);
        if (z.x < z.y) {
            temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - Menger_Scale_Offset); // same space transform as tetrahedron
        if (z.z < -Menger_Z_thing * MengerOffset.z * (MengerScale - Menger_Scale_Offset))
        {
            z.z += MengerOffset.z * (MengerScale - Menger_Scale_Offset);
        }
        dr = dr * abs(MengerScale) + 1.f;// dr_offset_Menger;


        // i removed an additional menger step bc the bulb is already here 
    }
    float fractal = length(z) / abs(dr);// +dr_offset_final);
    float result = max(fractal, -dist_to_XY_plane);
    result = max(result, -dist_to_XZ_plane);
    return result;
}



__device__ inline float intersect(float3 ro, float3 rd, float step_size, float min_distance, float& iter, Params params)
{
    float res;
    float t = 0.f;
    iter = max_iter;
    for (float i = 0.f; i < max_iter; i += 1.f) {
        float3 p = ro + rd * t;
        res = DE(p, params);
        if (res < min_distance * t || res > 20.f) {
            iter = i;
            break;
        }
        t += res * step_size;
    }
    if (res > 20.f) t = -1.f;
    return t;
}

__device__ inline float ambientOcclusion(float3 p, float3 n, Params params) {
    float stepSize = 0.012f;
    float t = stepSize;
    float oc = 0.0f;
    for (int i = 0; i < 12; i++) {
        float d = DE(p + n * t, params);
        oc += t - d;
        t += stepSize;
    }
    return clamp(oc, 0.0f, 1.0f);
}

__device__ inline float3 normal(float3 p, float EPS, Params params)
{
    float3 e0 = make_float3(EPS, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, EPS, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, EPS);
    float3 n = n = normalize(make_float3(
        DE(p + e0, params) - DE(p - e0, params),
        DE(p + e1, params) - DE(p - e1, params),
        DE(p + e2, params) - DE(p - e2, params)));
    return n;
}


__device__ inline float softshadow(float3 ro, float3 rd, float k, Params params) {
    float akuma = 1.0f, h = 0.0f;
    float t = 0.01f;
    for (int i = 0; i < 50; ++i) {
        h = DE(ro + rd * t, params);
        if (h < 0.001f)return 0.02f;
        akuma = min(akuma, k * h / t);
        t += clamp(h, 0.01f, 2.0f);
    }
    return akuma;
}


__device__ inline float3 lighting(float3 p, float3 rd, float3 light_dir, float iter, Technicals tech, Params params, unsigned int iTime)
{
    float3 n = normal(p, tech.EPS, params);
    float3 col;
    //   A M B I E N T   O C C L U S I O N 
#if defined(AMBIENT_OCCLUSION)
    float fake = iter / max_iter;
    float fakeAmb = exp(-fake * fake * 9.0f);
    float amb = ambientOcclusion(p, n, params);
    col = make_float3(mix(1.0f, 0.125f, pow(amb, 3.0f))) * make_float3(fakeAmb) * bone;
#endif
#if defined(RAINBOW)
    //   Rainbow Normals
    col = make_float3(1.f) - abs(n); // set normal as color with dark edges
    float y = mix(.45f, 1.2f, 0.45f);//pow(smoothstep(0.,1.,.75-dir.y),2.))*(1.-sb*.5); // gradient sky
    //col = mix(make_float3(1.f, .9f, .3f), col, exp(-.004));// distant fading to sun color
    col = pow(col, make_float3(GAMMA)) * BRIGHTNESS;
    col = mix(make_float3(length(col)), col, SATURATION);
    col *= make_float3(1.f, .9f, .85f);
#endif
#if defined(RYU)
    //   R Y U   L I G H T 
    float3 l1_dir = light_dir;
    //float3 l1_dir = normalize(make_float3(0.8f, 0.8f, 0.4f));
    float3 l1_col = 0.3f * make_float3(1.5f, 1.69f, 0.79f);
    float3 l2_dir = normalize(make_float3(-0.8f, 0.5f, 0.3f));
    float3 l2_col = make_float3(0.89f, 0.99f, 1.3f);

    float shadow = softshadow(p, l1_dir, 10.0f, params);
    float dif1 = max(0.0f, dot(n, l1_dir));
    float dif2 = max(0.0f, dot(n, l2_dir));
    float bac1 = max(0.3f + 0.7f * dot(make_float3(-l1_dir.x, -1.0f, -l1_dir.z), n), 0.0f);
    float bac2 = max(0.2f + 0.8f * dot(make_float3(-l2_dir.x, -1.0f, -l2_dir.z), n), 0.0f);
    float spe = max(0.0f, pow(clamp(dot(l1_dir, reflect(rd, n)), 0.0f, 1.0f), 10.0f));

    col = 5.5f * l1_col * dif1 * shadow;
    col += 1.1f * l2_col * dif2;
    col += 0.3f * bac1 * l1_col;
    col += 0.3f * bac2 * l2_col;
    col += 1.0f * spe;
#endif
#if defined(SEV)
    //   Rainbow Normals
    float3 rainbow_col = make_float3(1.f) - abs(n); // set normal as color with dark edges
    float y = mix(.45f, 1.2f, 0.45f);//pow(smoothstep(0.,1.,.75-dir.y),2.))*(1.-sb*.5); // gradient sky
    //col = mix(make_float3(1.f, .9f, .3f), col, exp(-.004));// distant fading to sun color
    rainbow_col = pow(rainbow_col, make_float3(GAMMA)) * BRIGHTNESS;
    rainbow_col = mix(make_float3(length(rainbow_col)), rainbow_col, SATURATION);
    rainbow_col *= make_float3(1.f, .9f, .85f);
    // light from AOC 
    // - diffuse light with hard shadow. doesnt work bc my fractal says everythign is in shadow :(
    float3 light_pos = make_float3(0.f, 10.f, -10.f);
    /* AOC SH ADOW - need to find out why its not working as well as soft shadow!!!
    // rotate light_pos
    float angle = degreesToRadians(0.f);
    float temp1 = light_pos.x;
    float temp2 = light_pos.y;
    light_pos.x = temp1 * cos(angle) + temp2 * sin(angle);
    light_pos.y = -temp1 * sin(angle) + temp2 * cos(angle);

    float3 l = normalize(light_pos - p);
    float dif = clamp(dot(n, l), 0.f, 1.f);
    float d = intersect(p + n * tech.min_distance*20.f, l, tech.step_size, tech.min_distance, iter, params);
    if (d < length(light_pos - p)) dif *= 0.1f; // the light is darker in the shadow
    float shadow = softshadow(p, light_pos-p, 10.0f, params);
    col *= dif * make_float3(2.f,1.f,1.5f);
    */
    //   R Y U   L I G H T 
    float3 l1_dir = light_pos - p;
    float3 l1_col = rainbow_col;

    float dif1 = max(0.0f, dot(n, l1_dir));
    float bac1 = max(0.3f + 0.7f * dot(make_float3(-l1_dir.x, -1.0f, -l1_dir.z), n), 0.0f);
    float spe = max(0.0f, pow(clamp(dot(l1_dir, reflect(rd, n)), 0.0f, 1.0f), 10.0f));

    float shadow = softshadow(p, light_pos - p, 10.0f, params);

    col = .1f * l1_col * dif1 * shadow;
    col += 0.3f * bac1 * l1_col;
    col += 0.3f * spe;
#endif
    return col;
}


__global__ void kernel(uchar4* map, unsigned int iTime, float3 _position, float3 _lookat,
    Technicals tech, Params params)
{
    //   C U R A N D 
    // CUDA's random number library uses curandState_t to keep track of the seed value
    // we will store a random state for every thread  
    curandState_t state;
    // we have to initialize the state 
    curand_init(0, // the seed controls the sequence of random values that are produced 
        0, // the sequence number is only important with multiple cores 
        0, // the offset is how much extra we advance in the sequence for each call, can be 0 
        &state);
    // curand works like rand - except that it takes a state as a parameter 


    int num_samples = 1;

    //  UV coords
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // ranges from 0 to 768
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // ranges from 0 to 512
    int idx = ix + iy * WIDTH;//blockDim.x * gridDim.x; 
    {
        float3 final_col = make_float3(0.f);
        for (int i = 0; i < num_samples; i++)
        {
            //   C O O R D I N A T E S 
            float qx = (((float)ix) + curand_uniform(&state)) / WIDTH; // ranges from 0 to 1
            float qy = (((float)iy) + curand_uniform(&state)) / (HEIGHT - HEIGHT_OFFSET); // ranges from 0 to 1
            //float qx = ((float)ix) / WIDTH; // ranges from 0 to 1
            //float qy = ((float)iy) / (HEIGHT - HEIGHT_OFFSET); // ranges from 0 to 1
            float uvx = ((qx - 0.5f) * 2.0f) * (((float)WIDTH) / (float)(HEIGHT - HEIGHT_OFFSET));  // range from -1 to 1
            float uvy = ((qy - 0.5f) * 2.0f); // range from -1 to 1


            //   C A M E R A 
            float3 lookat = _lookat;
            float3 ro = _position;
            float3 f = normalize(lookat - ro);
            float3 s = normalize(cross(f, make_float3(0.0f, 1.0f, 0.0f)));
            float3 u = normalize(cross(s, f));
            float3 rd = normalize(uvx * s + uvy * u + 2.8f * f);  // transform from view to world


            //   B A C K G R O U N D
            float3 bg = mix(bone * 0.5f, bone, smoothstep(-1.0f, 1.0f, uvy));
            float3 col = bg;
            float3 p = ro;


            //   R A Y M A R C H 
            float iter = 0.f;// # of steps 
            float t = intersect(ro, rd, tech.step_size, tech.min_distance, iter, params);


            //   L I G H T I N G 
            if (t > -0.5) {
                p = ro + t * rd;
                col = lighting(p, rd, normalize(make_float3(0.8f, 0.8f, 0.4f)), iter, tech, params, iTime);
                col = mix(col, bg, 1.0 - exp(-0.001 * t * t));
            }
#if defined(AMBIENT_OCCLUSION) || defined(RYU) 
            col = pow(clamp(col, 0.0f, 1.0f), make_float3(0.45f)); // P O S T 
            col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);  // contrast
            col = mix(col, make_float3(dot(col, make_float3(0.33f))), -0.5f);  // satuation
            col *= 0.5f + 0.5f * pow(19.0f * qx * qy * (1.0f - qx) * (1.0f - qy), 0.7f);  // vigneting
#endif
            col = clamp(col, 0.f, 1.f); // there is a bit of overflow somewhere but this fixes it.
            final_col += (col / float(num_samples));
        }
//#if defined(DEBUG) doesnt work here bc iter out of range 
//        map[idx].x = (unsigned char)iter * 2;
//        map[idx].y = (unsigned char)0;
//        map[idx].z = (unsigned char)0;
//        map[idx].w = (unsigned char)255;
//#else
        map[idx].x = (unsigned char)(255.0f * final_col.x);
        map[idx].y = (unsigned char)(255.0f * final_col.y);
        map[idx].z = (unsigned char)(255.0f * final_col.z);
        map[idx].w = (unsigned char)255;
//#endif
    }
}


void Camera::KernelLauncher()
{
    static unsigned int iTime = 0;
    iTime++;
    uchar4* dev_map;
    dim3 threads(8, 8);
    dim3 grids(WIDTH / 8, (HEIGHT - HEIGHT_OFFSET) / 8); // 96 x 64
    gpuErrchk(cudaGraphicsMapResources(1, &cudapbo, NULL));
    gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&dev_map, NULL, cudapbo));
    kernel << <grids, threads >> > (dev_map, iTime, position, lookat, tech, params);
    //gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));
}
