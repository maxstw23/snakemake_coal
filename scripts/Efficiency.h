#ifndef Efficiency_hh
#define Efficiency_hh

#include <string>

class Efficiency
{
    private:
    static const float P0_pip[9];
    static const float P1_pip[9];
    static const float P2_pip[9];
    static const float P3_pip[9];
    static const float P4_pip[9];
    static const float P0_pim[9];
    static const float P1_pim[9];
    static const float P2_pim[9];
    static const float P3_pim[9];
    static const float P4_pim[9];
    static const float P0_Kp[9];
    static const float P1_Kp[9];
    static const float P2_Kp[9];
    static const float P3_Kp[9];
    static const float P4_Kp[9];
    static const float P0_Km[9];
    static const float P1_Km[9];
    static const float P2_Km[9];
    static const float P3_Km[9];
    static const float P4_Km[9];
    static const float P0_P[9];
    static const float P1_P[9];
    static const float P2_P[9];
    static const float P3_P[9];
    static const float P4_P[9];
    static const float P0_AP[9];
    static const float P1_AP[9];
    static const float P2_AP[9];
    static const float P3_AP[9];
    static const float P4_AP[9];

    static const float P0_Kp_2D[9];
    static const float P1_Kp_2D[9];
    static const float P2_Kp_2D[9];
    static const float P3_Kp_2D[9];
    static const float P4_Kp_2D[9];
    static const float P5_Kp_2D[9];
    static const float P6_Kp_2D[9];
    static const float P0_Km_2D[9];
    static const float P1_Km_2D[9];
    static const float P2_Km_2D[9];
    static const float P3_Km_2D[9];
    static const float P4_Km_2D[9];
    static const float P5_Km_2D[9];
    static const float P6_Km_2D[9];

    public:
    Efficiency() {}
    float GetEfficiency1D(float pT, int cent, std::string particle);
    float GetEfficiency2D(float pT, float eta, int cent, std::string particle);
};

#endif