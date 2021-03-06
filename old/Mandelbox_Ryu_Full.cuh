#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// ------------------------------------------------ M A N D E L B O X   R Y U ------------------
// ---------------------------------------------------------------------------------------------

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


__device__ inline float softshadow(float3 ro, float3 rd, float k) {
    float akuma = 1.0f, h = 0.0f;
    float t = 0.01f;
    for (int i = 0; i < 50; ++i) {
        h = DE(ro + rd * t);
        if (h < 0.001f)return 0.02f;
        akuma = min(akuma, k * h / t);
        t += clamp(h, 0.01f, 2.0f);
    }
    return akuma;
}


__device__ inline float4 intersect(float3 ro, float3 rd, float step_size)
{
    float d = 0.f;//
    float steps = 0.f;//

    float t = 0.f;//
    float td = 0.f;
    float min_d = 1.f;
    for (; steps < 128.f; steps += 1.f) 
    {
        d = DE(ro + rd * t);//
        if (d < 0.001f * t || d > 20.f)  break;//
        t += d *step_size;//
        td += d;
        float sharpness = 10.f; 
        min_d = min(min_d, sharpness * d / td);
    }
    if (d > 20.f) t = -1.f;//
    return make_float4(t,steps,0.f,min_d);//
}


__device__ inline float3 lighting(float3 p, float3 rd, float ps, float3 light_dir) {
    float3 l1_dir = light_dir;
    //float3 l1_dir = normalize(make_float3(0.8f, 0.8f, 0.4f));
    float3 l1_col = 0.3f * make_float3(1.5f, 1.69f, 0.79f);
    float3 l2_dir = normalize(make_float3(-0.8f, 0.5f, 0.3f));
    float3 l2_col = make_float3(0.89f, 0.99f, 1.3f);

    float3 e0 = make_float3(0.5f * ps, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, 0.5f * ps, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, 0.5f * ps);
    float3 n = normalize(make_float3(
        DE(p + e0) - DE(p - e0),
        DE(p + e1) - DE(p - e1),
        DE(p + e2) - DE(p - e2)));

    float shadow = 0.0f;// softshadow(p, l1_dir, 10.0f);

    float dif1 = max(0.0f, dot(n, l1_dir));
    float dif2 = max(0.0f, dot(n, l2_dir));
    float bac1 = max(0.3f + 0.7f * dot(make_float3(-l1_dir.x, -1.0f, -l1_dir.z), n), 0.0f);
    float bac2 = max(0.2f + 0.8f * dot(make_float3(-l2_dir.x, -1.0f, -l2_dir.z), n), 0.0f);
    float spe = max(0.0f, pow(clamp(dot(l1_dir, reflect(rd, n)), 0.0f, 1.0f), 10.0f));

    float3 col = 5.5f * l1_col * dif1 * shadow;
    col += 1.1f * l2_col * dif2;
    col += 0.3f * bac1 * l1_col;
    col += 0.3f * bac2 * l2_col;
    col += 1.0f * spe;

    //float t = mod(p.y + 0.1 * texture(iChannel0, p.xz).x - time * 150.0, 5.0);
    //col = mix(col, make_float3(6.0, 6.0, 8.0),
    //    pow(smoothstep(0.0, .3, t) * smoothstep(0.6, .3, t), 15.0));;
    return col;
}

__device__ inline float3 post(float3 col, float2 q) {
    // post
    col = pow(clamp(col, 0.0f, 1.0f), make_float3(0.45f,0.45f,0.45f));
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

    // camera
    //float stime, ctime, time;
    //stime = sin(iTime * 0.1f);
    //ctime = cos(iTime * 0.1f);
    //time = iTime * 0.01f;
    float3 ta = _lookat;// make_float3(0.0, 0.0, 0.0);
    float3 ro = _position;//make_float3(3.0 * stime, 2.0 * ctime, 5.0 + 1.0 * stime);
    float3 cf = normalize(ta - ro);
    float3 cs = normalize(cross(cf, make_float3(0.0f, 1.0f, 0.0f)));
    float3 cu = normalize(cross(cs, cf));
    float3 rd = normalize(uvx * cs + uvy * cu + 1.8f * cf);  // transform from view to world

    float3 mtl = make_float3(1.0f, 1.3f, 1.23f) * 0.8f;

    float4 t4 = intersect(ro, rd, step_size);
    float t = t4.x;
    float3 bg = make_float3(0.0f);
    float3 col = bg;
    float3 p = ro;
    
    if (t > -0.5f) {
        p = ro + t * rd;
        col = lighting(p, rd, EPS, _position - _lookat) * mtl * 0.2f;
        col = mix(col, bg, 1.0f - exp(-0.001f * t * t));
    }
    /*   G L O W 
    float m = t4.w;
    float glow = (1.0f - m) * (1.0f - m) * .2f;// GLOW_COLOR_DELTA;
    col.x += glow;
    */
    //col.y += glow;
    //col.z += 0.7f*glow;
    //float a = 1.0 / (1.0 + t4.y * 40.f);
    //col += (1.0 - a) * .2f;
    //float fake = 2.f*t4.y / 128.f; // THEse THREE LINES LOOK REALLY FUCKING NICE 
    //float fakeAmb = exp(-fake * fake * 9.0);
    //col += fakeAmb;

    col = post(col, make_float2(qx,qy));

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






