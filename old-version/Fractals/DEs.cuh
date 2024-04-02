#include "Common.cuh"
#include "Camera.h"



__device__ inline float DE(float3 z, Params params)
{
    /*
    // 17 floats for Hybrid 1
    float3 MandelboxOffset          = make_float3(0.f); // TODO: not sure if this can / should be changed
    float3 MengerOffset             = make_float3(1.0f); // menger params
    float MengerScale               = 3.475f;
    float box_mult                  = 2.0f; // mandelbox params
    float FixedR2                   = 1.f;
    float MinR2                     = 0.f;
    float fold                      = 1.65f;
    float MandelboxScale            = 1.2f;
    float Menger_Scale_Offset       = 1.f;
    float Menger_Z_thing            = 0.5f;
    float dr_offset_Mandel          = 1.f; // technical, step scale bonus accuracy
    float dr_offset_Menger          = 1.f;
    float dr_offset_final           = 1.f; // this one for real does nothing
    */
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
    float dr_offset_Menger = params.p[15];
    float dr_offset_final = params.p[16];


    MandelboxOffset = z + MandelboxOffset;
    const int Iterations = 4;
    float dr = 1.0f;
    float r2, temp; // local

    for (int n = 0; n < Iterations; n++)
    {
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
        dr = dr * abs(MengerScale) + dr_offset_Menger;


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
        // i removed an additional menger step bc the bulb is already here 
    }
    return length(z) / abs(dr + dr_offset_final);
}


__device__ inline float DE_Mandelbox(float3 z, Params params)
{
    float3 MandelboxOffset = make_float3(params.p[0], params.p[1], params.p[2]);
    float box_mult = params.p[7];
    float FixedR2 = params.p[8];
    float MinR2 = params.p[9];
    float fold = params.p[10];
    float MandelboxScale = params.p[11];

    float3 offset = z;
    float dr = 1.0f;
    for (int n = 0; n < 15; ++n) {
        z = clamp(z, -fold, fold) * box_mult - z; // box fold

        float r2 = dot(z, z); // sphere fold 
        if (r2 < MinR2) {
            float temp = (FixedR2 / MinR2);
            z *= temp;
            dr *= temp;
        }
        else if (r2 < FixedR2) {
            float temp = (FixedR2 / r2);
            z *= temp;
            dr *= temp;
        }
        z = MandelboxScale * z + offset;
        dr = dr * abs(MandelboxScale) + 1.0f;
    }
    float r = length(z);
    return r / abs(dr);
}


__device__ inline float DE_OctKoch(float3 z, Params params)
{
    float plane1 = params.p[0];
    float plane2 = params.p[1];
    float plane3 = params.p[2];
    float plane4 = params.p[3];
    float Scale = params.p[4];
    float3 CScale = make_float3(params.p[5], params.p[6], params.p[7]);
    float Foldx = params.p[8];
    float Subx = params.p[9];
    float FoldMul = params.p[10];
    float ScaleOffset1 = params.p[11];
    float ScaleOffset2 = params.p[12];


    float dr = 1.f; // CORR 
    bool test1, test2; float tmp; // locals 
    for (int n = 0; n < 17; ++n) {
        z = abs(z);

        test1 = (z.x - z.z < plane1);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x - z.y < plane2);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x + z.z < plane3);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;
        test1 = (z.x + z.y < plane4);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;


        z.x = z.x - Subx * (Scale - ScaleOffset1);
        z.x = clamp(z.x, -Foldx, Foldx) * FoldMul - z.x;
        //z.x = abs(z.x);//optional

        test1 = (z.x - z.z < plane1);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x - z.y < plane2);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 + z.x * test1;
        z.x = z.x * test2 + tmp * test1;
        test1 = (z.x + z.z < plane3);
        test2 = !test1;
        tmp = z.z;
        z.z = z.z * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;
        test1 = (z.x + z.y < plane4);
        test2 = !test1;
        tmp = z.y;
        z.y = z.y * test2 - z.x * test1;
        z.x = z.x * test2 - tmp * test1;

        z -= CScale * (Scale - ScaleOffset2);
        z *= Scale;
        dr = dr * abs(Scale);//+ 1.; // CORR
    }
    return length(z) / abs(dr); // length is distance to shape, divided for correction!
}


__device__ inline float DE_MengerSmt(float3 z, Params params) {
    float sc = params.p[0];
    float sc1 = sc - params.p[1];
    float Cx = params.p[2];
    float Cy = params.p[3];
    float Cz = params.p[4];
    float s = params.p[5];
    float sc2 = sc1 / sc;
    float t = 9999999.0f;
    float dr = 1.f; // CORR
    for (int n = 0; n < 15; ++n) {
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
    }
    return length(z) / abs(dr); // length is distance to shape, divided for correction!
}


