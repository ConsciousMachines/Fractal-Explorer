// "Fractal Cartoon" - former "DE edge detection" by Kali
// There are no lights and no AO, only color by normals and dark edges.

#include "../Common.cuh"
#include "../Camera.h"

cudaGraphicsResource * cudapbo;

#define BRIGHTNESS 1.2
#define GAMMA 1.4
#define SATURATION .65
#define detail .001

#define M_PI 3.14159265359f
#define degreesToRadians(angleDegrees) (angleDegrees * M_PI / 180.0f)

// Distance function
__device__ inline float de(float3 pos) {
	float hid = 0.;
	float3 tpos = pos;
	tpos.z=fabsf(3.-fmodf(tpos.z,6.));
	float4 p = make_float4(tpos.x,tpos.y,tpos.z, 1.f);
	for (int i = 0; i < 4; i++) {// "Amazing Surface" fractal
		p.x = fabsf(p.x + 1.) - fabsf(p.x - 1.) - p.x;
		p.z = fabsf(p.z + 1.) - fabsf(p.z - 1.) - p.z;
		p.y -= .25;
		//p.xy*=rot(radians(35.));
		float angle = degreesToRadians(35.f);
		float temp1 = p.x;
		float temp2 = p.y;
		p.x = temp1 * cos(angle) + temp2 * sin(angle);
		p.y = -temp1 * sin(angle) + temp2 * cos(angle);
		auto soy = make_float3(p.x, p.y, p.z);
		p = p * 2. / clamp(dot(soy,soy), .2, 1.);
	}
	float fr = (length(fmaxf(make_float2(0.), make_float2(p.y, p.z) - 1.5)) - 1.) / p.w;
	float ro = max(fabsf(pos.x + 1.) - .3, pos.y - .35);
	ro = max(ro, -max(fabsf(pos.x + 1.) - .1, pos.y - .5));
	pos.z = fabsf(.25 - fmodf(pos.z, .5));
	ro = max(ro, -max(fabsf(pos.z) - .2, pos.y - .3));
	ro = max(ro, -max(fabsf(pos.z) - .01, -pos.y + .32));
	float d = min(fr, ro);
	return d;
}

// Calc normals, and here is edge detection, set to variable "edge"
__device__ inline float3 normal(float3 p, float& edge, float EPS) {
	float3 e0 = make_float3(EPS, 0.0f, 0.0f);
	float3 e1 = make_float3(0.0f, EPS, 0.0f);
	float3 e2 = make_float3(0.0f, 0.0f, EPS);
	float d1 = de(p - e0), d2 = de(p + e0);
	float d3 = de(p - e1), d4 = de(p + e1);
	float d5 = de(p - e2), d6 = de(p + e2);

	float d = de(p);
	edge = fabsf(d - 0.5 * (d2 + d1)) + fabsf(d - 0.5 * (d4 + d3)) + fabsf(d - 0.5 * (d6 + d5));//edge finder
	edge = min(1., powf(edge, .55) * 15.);
	return normalize(make_float3(d1 - d2, d3 - d4, d5 - d6));
}


__device__ inline float3 rainbow(float2 p)
{
	float q = max(p.x, -0.1);
	float s = sin(p.x * 7.0 + 70.0) * 0.08; // can animate this
	p.y += s;
	p.y *= 1.1;
	float3 c;
	if (p.x > 0.0) c = make_float3(0); else
		if (0.0 / 6.0 < p.y && p.y < 1.0 / 6.0) c = make_float3(255, 43, 14) / 255.0; else
			if (1.0 / 6.0 < p.y && p.y < 2.0 / 6.0) c = make_float3(255, 168, 6) / 255.0; else
				if (2.0 / 6.0 < p.y && p.y < 3.0 / 6.0) c = make_float3(255, 244, 0) / 255.0; else
					if (3.0 / 6.0 < p.y && p.y < 4.0 / 6.0) c = make_float3(51, 234, 5) / 255.0; else
						if (4.0 / 6.0 < p.y && p.y < 5.0 / 6.0) c = make_float3(8, 163, 255) / 255.0; else
							if (5.0 / 6.0 < p.y && p.y < 6.0 / 6.0) c = make_float3(122, 85, 255) / 255.0; else
								if (fabsf(p.y) - .05 < 0.0001) c = make_float3(0., 0., 0.); else
									if (fabsf(p.y - 1.) - .05 < 0.0001) c = make_float3(0., 0., 0.); else
										c = make_float3(0, 0, 0);
	c = mix(c, make_float3(length(c)), .15);
	return c;
}

// Raymarching and 2D graphics
__device__ inline float3 raymarch(float3 from, float3 dir,float& det)
{
	float edge = 0.;
	float3 p, norm;
	float d = 100.;
	float totdist = 0.;
	for (int i = 0; i < 150; i++) {
		if (d > det && totdist < 25.0) {
			p = from + totdist * dir;
			d = de(p);
			det = detail * exp(.13 * totdist);
			totdist += d;
		}
	}
	float3 col = make_float3(0.);
	p -= (det - d) * dir;
	norm = normal(p,edge, det);
	//col=1.-make_float3(edge); // show wireframe version
	col = (make_float3(1.) - fabs(norm)) * max(0., 1. - edge * .8); // set normal as color with dark edges
	totdist = clamp(totdist, 0., 26.);
	dir.y -= .02;
	float y = mix(.45, 1.2, 0.45);//pow(smoothstep(0.,1.,.75-dir.y),2.))*(1.-sb*.5); // gradient sky
	float3 backg = make_float3(0.2, 0., .4);

	col = mix(make_float3(1., .9, .3), col, exp(-.004 * totdist * totdist));// distant fading to sun color
	if (totdist > 25.) col = backg; // hit background
	col = pow(col, make_float3(GAMMA)) * BRIGHTNESS;
	col = mix(make_float3(length(col)), col, SATURATION);
	//col=1.-make_float3(length(col)); // for show only edges
	col *= make_float3(1., .9, .85);
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

	/*
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
	col = clamp(col, 0.f, 1.f); // sev genius input. there is a bit of overflow somewhere but this fixes it.
	*/
	
	float det = 1.f;
	float3 dir = normalize(make_float3(uvx * 1.8f, uvy * 1.8f, 1.f));//FOV
	float3 origin = _position;// make_float3(-1.f, .7f, 0.f);
	float3 from = origin;
	float3 col = raymarch(from, dir,det);
	
	
	
	map[idx].x = (unsigned char)(255.0f * col.x);
	map[idx].y = (unsigned char)(255.0f * col.y);
	map[idx].z = (unsigned char)(255.0f * col.z);
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