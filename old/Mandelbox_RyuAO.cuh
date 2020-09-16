#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// ------------------------------------------------ M A N D E L B O X   R Y U ------------------
// ---------------------------------------------------------------------------------------------

#define max_iter 120
#define bone make_float3(0.89f, 0.855f, 0.788f)

__device__ inline void sphere_fold(float3& z, float& dz) {
    float fixed_radius2 = 1.9f;
    float min_radius2 = 0.1f;
    float r2 = dot(z, z);
    if (r2 < min_radius2) {
        float temp = (fixed_radius2 / min_radius2);
        z *= temp;
        dz *= temp;
    }
    else if (r2 < fixed_radius2) {
        float temp = (fixed_radius2 / r2);
        z *= temp;
        dz *= temp;
    }
}

__device__ inline void box_fold(float3& z, float& dz) {
    float folding_limit = 1.0f;
    z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;
}


__device__ inline float DE(float3 z) {
    float scale = -2.8f;
    float3 offset = z;
    float dr = 1.0f;
    for (int n = 0; n < 15; ++n) {
        box_fold(z, dr);
        sphere_fold(z, dr);
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0f;
        //scale = -2.8 - 0.2 * stime;
    }
    float r = length(z);
    return r / abs(dr);
}


__device__ inline float intersect(float3 ro, float3 rd, float step_size, int& iter) {
    float res;
    float t = 0.f;
    iter = max_iter;
    for (int i = 0; i < max_iter; ++i) {
        float3 p = ro + rd * t;
        res = DE(p);
        if (res < 0.001f * t || res > 20.f) {
            iter = i;
            break;
        }
        t += res;
    }
    if (res > 20.f) t = -1.f;
    return t;
}


__device__ inline float ambientOcclusion(float3 p, float3 n) {
    float stepSize = 0.012f;
    float t = stepSize;
    float oc = 0.0f;
    for (int i = 0; i < 12; i++) {
        float d = DE(p + n * t);
        oc += t - d;
        t += stepSize;
    }
    return clamp(oc, 0.0f, 1.0f);
}

__device__ inline float3 normal(float3 p, float EPS) {

    float3 e0 = make_float3(EPS, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, EPS, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, EPS);
    float3 n = normalize(make_float3(
        DE(p + e0) - DE(p - e0),
        DE(p + e1) - DE(p - e1),
        DE(p + e2) - DE(p - e2)));
    return n;
}


__device__ inline float3 lighting(float3 p, float3 rd, float3 light_dir, int iter, float EPS) {
    float3 n = normal(p, EPS);

    //   A M B I E N T   O C C L U S I O N 
    float fake = float(iter) / float(max_iter);
    float fakeAmb = exp(-fake * fake * 9.0f);
    float amb = ambientOcclusion(p, n);
    float3 col_bone = make_float3(mix(1.0f, 0.125f, pow(amb, 3.0f))) * make_float3(fakeAmb) * bone;


    //   R Y U   L I G H T 
    float3 l1_dir = light_dir;
    //float3 l1_dir = normalize(make_float3(0.8f, 0.8f, 0.4f));
    float3 l1_col = 0.3f * make_float3(1.5f, 1.69f, 0.79f);
    float3 l2_dir = normalize(make_float3(-0.8f, 0.5f, 0.3f));
    float3 l2_col = make_float3(0.89f, 0.99f, 1.3f);

    float shadow = 1.0f;// softshadow(p, l1_dir, 10.0f);
    float dif1 = max(0.0f, dot(n, l1_dir));
    float dif2 = max(0.0f, dot(n, l2_dir));
    float bac1 = max(0.3f + 0.7f * dot(make_float3(-l1_dir.x, -1.0f, -l1_dir.z), n), 0.0f);
    float bac2 = max(0.2f + 0.8f * dot(make_float3(-l2_dir.x, -1.0f, -l2_dir.z), n), 0.0f);
    float spe = max(0.0f, pow(clamp(dot(l1_dir, reflect(rd, n)), 0.0f, 1.0f), 10.0f));

    float3 col_ryu = 5.5f * l1_col * dif1 * shadow;
    col_ryu += 1.1f * l2_col * dif2;
    col_ryu += 0.3f * bac1 * l1_col;
    col_ryu += 0.3f * bac2 * l2_col;
    col_ryu += 1.0f * spe;

    //   B O T H 
    // TODO: how to combine bone + ryu colors? no fucking clue. Perhaps multiply so that
    // AO works as basically a stencil. but AO's highs are lower! will need to experiment.
    // they might also be diff because of the post()
    // perhaps add an IMGUI slider. 
    //pow(col_ryu * col_bone, make_float3(2.f)); -> this gives  anice metallic hue, but too dark.
    //pow(2.*col_ryu * col_bone, make_float3(1.5f));

    //float3 my_bone = pow(col_bone, make_float3(2.2f));
    //col_bone *= 3.f; // put the thing around range 0~1
    //auto col_bone_sq = col_bone * col_bone; // make lows lower and highs higher
    //auto masked_ryu = col_bone_sq * col_ryu; // apply lows of bone to Ryu color
    //return pow(masked_ryu, make_float3(1.5f)); // makes it more metallic. still crap tho
    return col_ryu * col_bone;
}

__device__ inline float3 post(float3 col, float2 q) {
    col = pow(clamp(col, 0.0f, 1.0f), make_float3(0.45f)); // TODO: play w these values 
    col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);  // contrast
    col = mix(col, make_float3(dot(col, make_float3(0.33f))), -0.5f);  // satuation
    col *= 0.5f + 0.5f * pow(19.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.7f);  // vigneting
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

    float3 lookat = _lookat;
    float3 ro = _position;
    float3 f = normalize(lookat - ro);
    float3 s = cross(f, make_float3(0.0f, 1.0f, 0.0f));
    float3 u = cross(s, f);
    float3 rd = normalize(uvx * s + uvy * u + 2.8f * f); 



    float3 mtl = make_float3(1.0f, 1.3f, 1.23f) * 0.8f; // Ryu color
    float3 bg = mix(bone * 0.5f, bone, smoothstep(-1.0f, 1.0f, uvy)); // bone color
    float3 col = bg;
    float3 p = ro;
    int iter = 0;
    float t = intersect(ro, rd, EPS, iter);
    if (t > -0.5f) {
        p = ro + t * rd;
        col = lighting(p, rd, _position - _lookat, iter, EPS) * mtl * 0.2f;
        col = mix(col, bg, 1.0f - exp(-0.001f * t * t));
    }



    col = post(col, make_float2(qx, qy));
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






