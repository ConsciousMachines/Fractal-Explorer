#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource* cudapbo;

// -------------------------------------------- M A N D E L B O X   B O N E Y ------------------
// ---------------------------------------------------------------------------------------------

#define max_iter 120
#define bone make_double3(0.89, 0.855, 0.788)

inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// https://developer.download.nvidia.com/cg/pow.html
__device__ inline double3 pow(double3 x, double3 y)
{
    double3 rv;
    rv.x = pow(x.x, y.x);//exp(x.x * log(y.x));
    rv.y = pow(x.y, y.y);//exp(x.y * log(y.y));
    rv.z = pow(x.z, y.z);//exp(x.z * log(y.z));
    return rv;
}

inline __host__ __device__ void operator*=(double3& a, double3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}



// https://www.reddit.com/r/opengl/comments/6nghtj/glsl_mix_implementation_incorrect/
__device__ inline double3 mix(double3 v1, double3 v2, double a)
{
    double3 result;
    result.x = v1.x * (1. - a) + v2.x * a;
    result.y = v1.y * (1. - a) + v2.y * a;
    result.z = v1.z * (1. - a) + v2.z * a;
    return result;
}

inline __host__ __device__ void operator*=(double3& a, double b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}


__device__ inline double mix(double v1, double v2, double a)
{
    return v1 * (1. - a) + v2 * a;
}

inline __device__ __host__ double clamp(double f, double a, double b)
{
    return fmaxf(a, fminf(f, b));
}

inline __device__ __host__ double3 clamp(double3 v, double a, double b)
{
    return make_double3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __host__ __device__ double3 operator*(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator*(double b, double3 a)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(double3& a, double3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ double3 operator+(double3 a, double b)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ double3 operator+( double b, double3 a)
{
    return make_double3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ double length(double3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ double3 normalize(double3 v)
{
    double invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ double3 make_double3(double s)
{
    return make_double3(s, s, s);
}
inline __host__ __device__ double3 operator*(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ double3 operator-(double3 a, double b)
{
    return make_double3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ double3 operator-(double b, double3 a)
{
    return make_double3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ double3 cross(double3 a, double3 b)
{
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}



__device__ inline void sphere_fold(double3& z, double& dz) {
    double fixed_radius2 = 1.9;
    double min_radius2 = 0.1;
    double r2 = dot(z, z);
    if (r2 < min_radius2) {
        double temp = (fixed_radius2 / min_radius2);
        z *= temp;
        dz *= temp;
    }
    else if (r2 < fixed_radius2) {
        double temp = (fixed_radius2 / r2);
        z *= temp;
        dz *= temp;
    }
}

__device__ inline void box_fold(double3& z, double& dz) {
    double folding_limit = 1.0;
    z = clamp(z, -folding_limit, folding_limit) * 2.0 - z;
}

__device__ inline double DE(double3 z) {
    double scale = -2.8;
    double3 offset = z;
    double dr = 1.0;
    for (int n = 0; n < 15; ++n) {
        box_fold(z, dr);
        sphere_fold(z, dr);
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0;
        //scale = -2.8 - 0.2 * stime;
    }
    double r = length(z);
    return r / abs(dr);
}

__device__ inline double intersect(double3 ro, double3 rd, double step_size, int& iter) {
    double res;
    double t = 0.;
    iter = max_iter;
    for (int i = 0; i < max_iter; ++i) {
        double3 p = ro + rd * t;
        res = DE(p);
        if (res < 0.001 * t || res > 20.) {
            iter = i;
            break;
        }
        t += res;
    }
    if (res > 20.) t = -1.;
    return t;
}

__device__ inline double ambientOcclusion(double3 p, double3 n) {
    double stepSize = 0.012;
    double t = stepSize;
    double oc = 0.0;
    for (int i = 0; i < 12; i++) {
        double d = DE(p + n * t);
        oc += t - d;
        t += stepSize;
    }
    return clamp(oc, 0.0, 1.0);
}

__device__ inline double3 normal(double3 p, double EPS) {

    double3 e0 = make_double3(EPS, 0.0, 0.0);
    double3 e1 = make_double3(0.0, EPS, 0.0);
    double3 e2 = make_double3(0.0, 0.0, EPS);
    double3 n = normalize(make_double3(
        DE(p + e0) - DE(p - e0),
        DE(p + e1) - DE(p - e1),
        DE(p + e2) - DE(p - e2)));
    return n;
}

__device__ inline double3 lighting(double3 p, double3 rd, int iter, double EPS) {
    double3 n = normal(p, EPS);
    double fake = double(iter) / double(max_iter);
    double fakeAmb = exp(-fake * fake * 9.0);
    double amb = ambientOcclusion(p, n);

    double soy = pow(amb, 3.0);
    double boy = mix(1.0, 0.125, soy);
    double3 col = make_double3(boy);

    return col * make_double3(fakeAmb) * bone;
}

__device__ inline double3 post(double3 col, double2 q) {
    col = pow(clamp(col, 0.0, 1.0), make_double3(0.45));
    col = col * 0.6 + 0.4 * col * col * (3.0 - 2.0 * col);  // contrast
    col = mix(col, make_double3(dot(col, make_double3(0.33))), -0.5);  // satuation
    col *= 0.5 + 0.5 * pow(19.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.7);  // vigneting
    return col;
}

__global__ void kernel(uchar4* map, unsigned int iTime, double3 _position, double3 _lookat, double step_size, double EPS) {
    //  UV coords
    int ix = threadIdx.x + blockIdx.x * blockDim.x; // ranges from 0 to 768
    int iy = threadIdx.y + blockIdx.y * blockDim.y; // ranges from 0 to 512
    int idx = ix + iy * WIDTH;//blockDim.x * gridDim.x; 
    double qx = ((double)ix) / WIDTH; // ranges from 0 to 1
    double qy = ((double)iy) / HEIGHT; // ranges from 0 to 1
    double uvx = ((qx - 0.5) * 2.0) * (WIDTH / HEIGHT);  // range from -1 to 1
    double uvy = ((qy - 0.5) * 2.0); // range from -1 to 1


    // camera
    double3 lookat = _lookat;
    double3 ro = _position;
    double3 f = normalize(lookat - ro);
    double3 s = normalize(cross(f, make_double3(0.0, 1.0, 0.0)));
    double3 u = normalize(cross(s, f));
    double3 rd = normalize(uvx * s + uvy * u + 2.8 * f);  // transform from view to world


    // background
    double3 bg = mix(bone * 0.5, bone, smoothstep(-1.0, 1.0, uvy));
    double3 col = bg;
    double3 p = ro;
    int iter = 0;
    double t = intersect(ro, rd, EPS, iter);
    if (t > -0.5) {
        p = ro + t * rd;
        col = lighting(p, rd, iter, EPS);
        col = mix(col, bg, 1.0 - exp(-0.001 * t * t));
    }
    col = post(col, make_double2(qx, qy));
    col = clamp(col, 0., 1.); // sev genius input. there is a bit of overflow somewhere but this fixes it.
    map[idx].x = (unsigned char)(255.0 * col.x);//(unsigned char)(255.0f * uvx);//ix / 2;
    map[idx].y = (unsigned char)(255.0 * col.y);//(unsigned char)(255.0f * uvy);//iy / 2;
    map[idx].z = (unsigned char)(255.0 * col.z);//(unsigned char)iTime;
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

    auto pos = make_double3(camera_ptr->position.x, camera_ptr->position.y, camera_ptr->position.z);
    auto lok = make_double3(camera_ptr->lookat.x, camera_ptr->lookat.y, camera_ptr->lookat.z);

    kernel << <grids, threads >> > (dev_map, iTime, pos, lok, camera_ptr->step_size, camera_ptr->EPS);
    //gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGraphicsUnmapResources(1, &cudapbo, NULL));
}



