#pragma once
#include <cuda_runtime.h>
//////// kernel version ///////////////////////////////////////////

        // Calculates the IR band Rosseland mean opacity (local T) according to the
        // Freedman et al. (2014) fit and coefficents

__device__ void kernel_k_Ross_Freedman(double Tin, double Pin, double met, double& k_IR) {
    // dependcies
    //// powl from math
    //// log10l from math        
    //// atan from math
    //// onedivpi -> namespace constants::onedivpi

    // Input:
    // T - Local gas temperature [K]
    // P - Local gas pressure [pa]
    // met - Local metallicity [M/H] (log10l from solar, solar [M/H] = 0.0)

    // Call by reference (Input&Output):
    // k_IR - IR band Rosseland mean opacity [m2 kg-1]

    const double pi = atan((double)(1)) * 4;
    const double onedivpi = 1.0 / pi;

    // Coefficent parameters for Freedman et al. (2014) table fit
    double c1 = 10.602;
    double c2 = 2.882;
    double c3 = 6.09e-15;
    double c4 = 2.954;
    double c5 = -2.526;
    double c6 = 0.843;
    double c7 = -5.490;
    double c8_l = -14.051, c8_h = 82.241;
    double c9_l = 3.055, c9_h = -55.456;
    double c10_l = 0.024, c10_h = 8.754;
    double c11_l = 1.877, c11_h = 0.7048;
    double c12_l = -0.445, c12_h = -0.0414;
    double c13_l = 0.8321, c13_h = 0.8321;

    // work variables
    double k_lowP;
    double k_hiP;
    double T;
    double P;
    double Tl10;
    double Pl10;

    // start operations

    T = Tin;
    P = Pin * ((double)10.0); // Convert to dyne cm-2


    Tl10 = log10((double)(T));
    Pl10 = log10((double)(P));

    // Low pressure expression
    k_lowP = c1 * atan((double)(Tl10 - c2)) -
        (c3 / (Pl10 + c4)) * exp((double)(pow((double)(Tl10 - c5), 2.0))) + c6 * met + c7;

    // De log10l
    k_lowP = pow((double)(10.0), k_lowP);

    // Temperature split for coefficents = 800 K
    if (T <= 800.0)
    {
        k_hiP = c8_l + c9_l * Tl10 + c10_l * pow((double)(Tl10), 2.0) +
            Pl10 * (c11_l + c12_l * Tl10) +
            c13_l * met * (0.5 + onedivpi * atan((double)((Tl10 - ((double)2.5)) / (double)0.2)));
    }
    else
    {
        k_hiP = c8_h + c9_h * Tl10 +
            c10_h * pow((double)(Tl10), 2.0) + Pl10 * (c11_h + c12_h * Tl10) +
            c13_h * met * (0.5 + onedivpi * atan((double)((Tl10 - ((double)2.5)) / (double)0.2)));
    }

    // De log10l
    k_hiP = pow((double)(10.0), k_hiP);

    // Total Rosseland mean opacity - converted to m2 kg-1
    k_IR = (k_lowP + k_hiP) / ((double)10.0);

    // Avoid divergence in fit for large values
    if (k_IR > 1.0e10)
    {
        k_IR = 1.0e10;
    }
}
