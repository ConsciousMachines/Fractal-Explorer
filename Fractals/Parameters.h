#include "Camera.h"



void reset_parameters_Hybrid1(Camera& camera)
{
    camera.params.p[0] = 0.f; // MandelboxOffset.x
    camera.params.p[1] = 0.f; // MandelboxOffset.y
    camera.params.p[2] = 0.f; // MandelboxOffset.z
    camera.params.p[3] = 1.f; // MengerOffset.x
    camera.params.p[4] = 1.f; // MengerOffset.y
    camera.params.p[5] = 1.f; // MengerOffset.z
    camera.params.p[6] = 3.475f; // MengerScale
    camera.params.p[7] = 2.f; // box_mult
    camera.params.p[8] = 1.f; // FixedR2
    camera.params.p[9] = 0.f; // MinR2
    camera.params.p[10] = 1.65f; // fold
    camera.params.p[11] = 1.2f; // MandelboxScale
    camera.params.p[12] = 1.f; // Menger_Scale_Offset
    camera.params.p[13] = 0.5f; // Menger_Z_thing
    camera.params.p[14] = 1.f; // dr_offset_Mandel
    camera.params.p[15] = 10.f; // dr_offset_Menger
    camera.params.p[16] = -4.f; // dr_offset_final
}


void render_options_Hybrid1(Camera& camera) // Render the ImGui sliders :D
{
    ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    ImGui::Begin("Mandelbox ----- V/B-> inc/dec min_dist, N/M-> inc/dec EPS, K/L-> inc/dec step_size");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    if (ImGui::Button("Reset"))
    {
        reset_parameters_Hybrid1(camera);
    }
    ImGui::SliderFloat("Mandelbox Offset X", &camera.params.p[0], -10.f, 10.f);
    ImGui::SliderFloat("Mandelbox Offset Y", &camera.params.p[1], -10.f, 10.f);
    ImGui::SliderFloat("Mandelbox Offset Z", &camera.params.p[2], -10.f, 10.f);
    ImGui::SliderFloat("Menger Offset X", &camera.params.p[3], -10.f, 10.f);
    ImGui::SliderFloat("Menger Offset Y", &camera.params.p[4], -10.f, 10.f);
    ImGui::SliderFloat("Menger Offset Z", &camera.params.p[5], -10.f, 10.f);
    ImGui::SliderFloat("Menger Scale", &camera.params.p[6], -5.f, 5.f);
    ImGui::SliderFloat("Box Mult", &camera.params.p[7], -5.f, 5.f);
    ImGui::SliderFloat("Fixed R2", &camera.params.p[8], -5.f, 5.f);
    ImGui::SliderFloat("Min R2", &camera.params.p[9], -5.f, 5.f);
    ImGui::SliderFloat("Fold", &camera.params.p[10], -5.f, 5.f);
    ImGui::SliderFloat("Mandelbox Scale", &camera.params.p[11], -5.f, 5.f);
    ImGui::SliderFloat("Menger Scale Offset", &camera.params.p[12], -10.f, 10.f);
    ImGui::SliderFloat("Menger Z thing", &camera.params.p[13], -5.f, 5.f);
    ImGui::SliderFloat("dr_offset_Mandel", &camera.params.p[14], 0.f, 5.f);
    ImGui::SliderFloat("XZ cut plane", &camera.params.p[15], -10.f, 10.f);
    ImGui::SliderFloat("XY cut plane", &camera.params.p[16], -10.f, 10.f);
    ImGui::End();
}


