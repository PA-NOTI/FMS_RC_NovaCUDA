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



__device__ void Ray_dry_adj(int nlay, int nlay1, double t_step, double kappa,
    double* Tl, double* pl,
    double* pe, double*& dT_conv, double* Tl_cc__df_l, double* d_p__df_l) {
    // dependcies
   //// main_parameters::nlay  -> "FMS_RC_para_&_const.cpp" 
   //// powl -> math    
   //// logl10 -> math
   //// expl -> math

   // Input:
   // 

   // Call by reference (Input & Output):
   // 

    // constants & parameters

    int itermax = 5;
    const double small = 1e-6;

    // work variables
    int i, iter;
    bool did_adj;
    double pfact, Tbar;
    double condi;

    // start operations

    for (i = 0; i < nlay; i++)
    {
        Tl_cc__df_l[i] = Tl[i];
        d_p__df_l[i] = pe[i + 1] - pe[i];

    }

    for (iter = 0; iter < itermax; iter++)
    {
        did_adj = false;

        // Downward pass
        for (i = 0; i < nlay - 1; i++)
        {
            pfact = pow((double)(pl[i] / pl[i + 1]), kappa);
            condi = (Tl_cc__df_l[i + 1] * pfact - small);


            if (Tl_cc__df_l[i] < condi) {
                Tbar = (d_p__df_l[i] * Tl_cc__df_l[i] + d_p__df_l[i + 1] * Tl_cc__df_l[i + 1]) /
                    (d_p__df_l[i] + d_p__df_l[i + 1]);

                Tl_cc__df_l[i + 1] = (d_p__df_l[i] + d_p__df_l[i + 1]) * Tbar /
                    (d_p__df_l[i + 1] + pfact * d_p__df_l[i]);

                Tl_cc__df_l[i] = Tl_cc__df_l[i + 1] * pfact;


                did_adj = true;
            }
        }

        // Upward pass
        for (i = nlay - 2; i > -1; i--) {
            pfact = pow((double)(pl[i] / pl[i + 1]), kappa);
            condi = (Tl_cc__df_l[i + 1] * pfact - small);


            if (Tl_cc__df_l[i] < condi) {
                Tbar = (d_p__df_l[i] * Tl_cc__df_l[i] + d_p__df_l[i + 1] * Tl_cc__df_l[i + 1]) /
                    (d_p__df_l[i] + d_p__df_l[i + 1]);

                Tl_cc__df_l[i + 1] = (d_p__df_l[i] + d_p__df_l[i + 1]) * Tbar /
                    (d_p__df_l[i + 1] + pfact * d_p__df_l[i]);

                Tl_cc__df_l[i] = Tl_cc__df_l[i + 1] * pfact;

                did_adj = true;
            }
        }

        // ! If no adjustment required, exit the loop
        if (did_adj == false)
        {
            break;
        }
    }

    // Change in temperature is Tl_cc - Tl
    // adjust on timescale of 1 timestep
    for (i = 0; i < nlay; i++)
    {
        dT_conv[i] = (Tl_cc__df_l[i] - Tl[i]) / t_step;
    }

}

///////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////

__device__  void linear_log_interp(double xval, double x1, double x2, double y1, double y2, double& yval) {
    // dependcies
    //// powll from math
    //// log10f from math

    // work variables
    double lxval;
    double ly1;
    double ly2;
    double lx1;
    double lx2;
    double norm;

    // start operations
    lxval = log10((double)(xval));
    lx1 = log10((double)(x1));
    lx2 = log10((double)(x2));
    ly1 = log10((double)(y1));
    ly2 = log10((double)(y2));

    norm = ((double)1.0) / (lx2 - lx1);

    yval = pow((double)(10.0), ((ly1 * (lx2 - lxval) + ly2 * (lxval - lx1)) * norm));
}

///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

__device__ void tau_struct(int nlev, double grav,
    double* p_half, double* kRoss,
    int channel, double* tau_struc_e) {
    // dependencies
    //// nlay -> namespace main_parameters    
    //// nlay1 -> namespace main_parameters

    // work variables
    double tau_sum;
    double tau_lay;
    double delP;
    int k;

    // running sum of optical depth
    tau_sum = 0.0;

    // start operations
    //  Upper most tau_struc is given by some low pressure value (here 1e-9 bar = 1e-4 pa)
    //dP = (p_half(1) - 1e-4)
    //tau_lay = (kRoss(1) * dP) / grav
    //tau_sum = tau_sum + tau_lay
    tau_struc_e[0] = tau_sum;

    // Integrate from top to bottom    

    for (k = 0; k < nlev; k++)
    {
        // Pressure difference between layer edges
        delP = (p_half[k + 1] - p_half[k]);

        // Optical depth of layer assuming hydrostatic equilibirum
        tau_lay = (kRoss[channel * nlev + k] * delP) / grav;

        // Add to running sum
        tau_sum = tau_sum + tau_lay;

        // Optical depth structure is running sum
        tau_struc_e[k + 1] = tau_sum;
    }

}

///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

__device__  void sw_grey_down(int nlay1, double solar,
    double* solar_tau, double* sw_down__df_e, double mu) {
    // dependencies
    //// expll -> math

    // work variables
    int i;

    // start operations
    for (i = 0; i < nlay1; i++)
    {
        sw_down__df_e[i] = solar * mu * exp((double)(-solar_tau[i] / mu));
    }

}

///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

__device__  void lw_grey_updown_linear(int nlay, int nlay1,
    double* be__df_e, double* tau_IRe__df_e,
    double* lw_up__df_e, double* lw_down__df_e,
    double* dtau__dff_l, double* del__dff_l,
    double* edel__dff_l, double* e0i__dff_l, double* e1i__dff_l,
    double* Am__dff_l, double* Bm__dff_l,
    double* lw_up_g__dff_l, double* lw_down_g__dff_l) {
    // dependencies
    //// expll -> math
    //// main_parameters::nlay1    
    //// main_parameters::nlay
    //// constants::gauss_ng
    //// constants::twopi

    const double pi = atan((double)(1)) * 4;
    const double  twopi = 2.0 * pi;

    // Work variables and arrays
    int k, g;



    //Gauss quadrature variables
    const int gauss_ng = 2;
    double uarr[gauss_ng];
    double w[gauss_ng];

    uarr[0] = 0.21132487;
    uarr[1] = 0.78867513;
    w[0] = 0.5;
    w[1] = 0.5;

    for (k = 0; k < nlay; k++)
    {
        dtau__dff_l[k] = (tau_IRe__df_e[k + 1] - tau_IRe__df_e[k]);
    }

    // Zero the flux arrays
    for (k = 0; k < nlay1; k++)
    {
        lw_down__df_e[k] = 0.0;
        lw_up__df_e[k] = 0.0;
    }

    // Start loops to integrate in mu space
    for (g = 0; g < gauss_ng; g++)
    {
        // Prepare loop
        for (k = 0; k < nlay; k++)
        {
            // Olson & Kunasz (1987) parameters
            del__dff_l[k] = dtau__dff_l[k] / uarr[g];
            edel__dff_l[k] = exp((double)(-del__dff_l[k]));
            e0i__dff_l[k] = 1.0 - edel__dff_l[k];
            e1i__dff_l[k] = del__dff_l[k] - e0i__dff_l[k];

            Am__dff_l[k] = e0i__dff_l[k] - e1i__dff_l[k] / del__dff_l[k]; // Am[k] = Gp[k], just indexed differently
            Bm__dff_l[k] = e1i__dff_l[k] / del__dff_l[k]; // Bm[k] = Bp[k], just indexed differently

        }

        // Peform downward loop first
        // Top boundary condition
        lw_down_g__dff_l[0] = 0.0;
        for (k = 0; k < nlay; k++)
        {
            lw_down_g__dff_l[k + 1] = lw_down_g__dff_l[k] * edel__dff_l[k] + Am__dff_l[k] * be__df_e[k] + Bm__dff_l[k] * be__df_e[k + 1]; // TS intensity
        }


        // Peform upward loop
        // Lower boundary condition
        lw_up_g__dff_l[nlay1 - 1] = be__df_e[nlay1 - 1];
        for (k = nlay - 1; k > -1; k--)
        {
            lw_up_g__dff_l[k] = lw_up_g__dff_l[k + 1] * edel__dff_l[k] + Bm__dff_l[k] * be__df_e[k] + Am__dff_l[k] * be__df_e[k + 1]; // TS intensity
        }



        // Sum up flux arrays with Gauss weights and points
        for (k = 0; k < nlay1; k++)
        {
            lw_down__df_e[k] = lw_down__df_e[k] + lw_down_g__dff_l[k] * w[g] * uarr[g];
            lw_up__df_e[k] = lw_up__df_e[k] + lw_up_g__dff_l[k] * w[g] * uarr[g];
        }
    }

    for (k = 0; k < nlay1; k++)
    {
        lw_down__df_e[k] = twopi * lw_down__df_e[k];
        lw_up__df_e[k] = twopi * lw_up__df_e[k];
    }

}


///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

__device__  void lw_grey_updown_poly(int nlay, int nlay1, double* be__df_e,
    double* tau_IRe__df_e, double* lw_up__df_e,
    double* lw_down__df_e, double* dtau__dff_l, double* del__dff_l,
    double* edel__dff_l, double* e0i__dff_l, double* e1i__dff_l,
    double* e2i__dff_l, double* Am__dff_l, double* Bm__dff_l,
    double* Gm__dff_l, double* lw_up_g__dff_l, double* lw_down_g__dff_l) {
    // dependencies
    //// expll -> math
    //// powll -> math
    //// main_parameters::nlay1    
    //// main_parameters::nlay
    //// constants::gauss_ng
    //// constants::twopi
    const double pi = atan((double)(1)) * 4;
    const double  twopi = 2.0 * pi;

    // Work variables and arrays
    int k, g;

    //Gauss quadrature variables
    const int gauss_ng = 2;
    double uarr[gauss_ng];
    double w[gauss_ng];

    uarr[0] = 0.21132487;
    uarr[1] = 0.78867513;
    w[0] = 0.5;
    w[1] = 0.5;

    for (k = 0; k < nlay; k++)
    {
        dtau__dff_l[k] = (tau_IRe__df_e[k + 1] - tau_IRe__df_e[k]);
    }

    // Zero the flux arrays
    for (k = 0; k < nlay1; k++)
    {
        lw_up__df_e[k] = 0.0;
        lw_down__df_e[k] = 0.0;
    }


    // Start loops to integrate in mu space
    for (g = 0; g < gauss_ng; g++)
    {
        // Prepare loop
        for (k = 0; k < nlay; k++)
        {
            // Olson & Kunasz (1987) parameters
            del__dff_l[k] = dtau__dff_l[k] / uarr[g];
            edel__dff_l[k] = exp((double)(-del__dff_l[k]));
            e0i__dff_l[k] = ((double)(1.0)) - edel__dff_l[k];
            e1i__dff_l[k] = del__dff_l[k] - e0i__dff_l[k];
            e2i__dff_l[k] = pow((double)(del__dff_l[k]), 2) - 2.0 * e1i__dff_l[k];
        }

        for (k = 0; k < nlay; k++) {
            // For boundary conditions assume linear interpolation at edges
            if (k == 1 || k == nlay)
            {
                Am__dff_l[k] = e0i__dff_l[k] - e1i__dff_l[k] / del__dff_l[k]; // Am[k] = Gp[k], just indexed differently
                Bm__dff_l[k] = e1i__dff_l[k] / del__dff_l[k]; // Bm[k] = Bp[k], just indexed differently
                Gm__dff_l[k] = 0.0;// Gm(k) = Ap(k)
            }
            else
            {
                Am__dff_l[k] = e0i__dff_l[k] + (e2i__dff_l[k] - (del__dff_l[k + 1] + 2.0 * del__dff_l[k]) * e1i__dff_l[k]) / (del__dff_l[k] * (del__dff_l[k + 1] + del__dff_l[k])); // Am[k] = Gp[k], just indexed differently
                Bm__dff_l[k] = ((del__dff_l[k + 1] + del__dff_l[k]) * e1i__dff_l[k] - e2i__dff_l[k]) / (del__dff_l[k] * del__dff_l[k + 1]); // Bm[k] = Bp[k], just indexed differently
                Gm__dff_l[k] = (e2i__dff_l[k] - del__dff_l[k] * e1i__dff_l[k]) / (del__dff_l[k + 1] * (del__dff_l[k + 1] + del__dff_l[k])); // Gm[k] = Ap[k], just indexed differently
            }
        }

        // Peform downward loop first
        // Top boundary condition
        lw_down_g__dff_l[0] = 0.0;
        lw_down_g__dff_l[1] = lw_down_g__dff_l[0] * edel__dff_l[0] + Am__dff_l[0] * be__df_e[0] + Bm__dff_l[0] * be__df_e[1];
        for (k = 1; k < nlay - 1; k++)
        {
            lw_down_g__dff_l[k + 1] = lw_down_g__dff_l[k] * edel__dff_l[k] + Am__dff_l[k] * be__df_e[k] + Bm__dff_l[k] * be__df_e[k + 1] +
                Gm__dff_l[k] * be__df_e[k - 1]; // TS intensity
        }
        lw_down_g__dff_l[nlay1 - 1] = lw_down_g__dff_l[nlay - 1] * edel__dff_l[nlay - 1] + Am__dff_l[nlay - 1] * be__df_e[nlay - 1] + Bm__dff_l[nlay - 1] * be__df_e[nlay1 - 1];

        // Peform upward loop
        // Lower boundary condition
        lw_up_g__dff_l[nlay1 - 1] = be__df_e[nlay1 - 1];
        lw_up_g__dff_l[nlay - 1] = lw_up_g__dff_l[nlay1 - 1] * edel__dff_l[nlay - 1] + Bm__dff_l[nlay - 1] * be__df_e[nlay - 1] + Am__dff_l[nlay - 1] * be__df_e[nlay1 - 1];
        for (k = nlay - 2; k > 0; k--)
        {
            lw_up_g__dff_l[k] = lw_up_g__dff_l[k + 1] * edel__dff_l[k] + Gm__dff_l[k] * be__df_e[k - 1] + Bm__dff_l[k] * be__df_e[k] + Am__dff_l[k] * be__df_e[k + 1]; // TS intensity
        }
        lw_up_g__dff_l[0] = lw_up_g__dff_l[1] * edel__dff_l[0] + Bm__dff_l[0] * be__df_e[0] + Am__dff_l[0] * be__df_e[1];

        // Sum up flux arrays with Gauss weights and points
        for (k = 0; k < nlay1; k++)
        {
            lw_down__df_e[k] = lw_down__df_e[k] + lw_down_g__dff_l[k] * w[g] * uarr[g];
            lw_up__df_e[k] = lw_up__df_e[k] + lw_up_g__dff_l[k] * w[g] * uarr[g];
        }
    }

    for (k = 0; k < nlay1; k++)
    {
        lw_down__df_e[k] = twopi * lw_down__df_e[k];
        lw_up__df_e[k] = twopi * lw_up__df_e[k];
    }

}

///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////



    __device__ void Kitzmann_TS_noscatt(const int nlay, const int nlay1, double *Tl,
        double *pl, double *pe,
        double *k_V_l, double *k_IR_l,
        double *Beta_V, double *Beta, double *&net_F,
        double mu_s, double Finc, double Fint, double grav, double AB,

        double *tau_Ve__df_e, double *tau_IRe__df_e, double *Te__df_e, double *be__df_e, //Kitzman working variables
        double *sw_down__df_e, double *sw_down_b__df_e, double *sw_up__df_e,
        double *lw_down__df_e, double *lw_down_b__df_e,
        double *lw_up__df_e, double *lw_up_b__df_e,
        double *lw_net__df_e, double *sw_net__df_e,

        double *dtau__dff_l, double *del__dff_l, // lw_grey_updown_linear working variables
        double *edel__dff_l, double *e0i__dff_l, double *e1i__dff_l,
        double *Am__dff_l, double *Bm__dff_l,
        double *lw_up_g__dff_l, double *lw_down_g__dff_l) {
        // dependcies
        //// powll -> include math
        //// log10f -> include math
        //// nlay -> namespace main_parameters
        //// nlay1 -> namespace main_parameters
        //// linear_log_interp -> namespace Kitsmann
        //// tau_struct -> namespace Kitsmann
        //// sw_grey_down -> namespace Kitsmann
        //// lw_grey_updown_linear -> namespace Kitsmann
        //// (lw_grey_updown_poly) -> namespace Kitsmann

        const double pi = atan((double)(1)) * 4;
        const double  twopi = 2.0 * pi;
        const double StBC = 5.670374419e-8;

        


        // work variables
        //int i;
        //double Finc_B;


       
        // start operation

        // Find temperature at layer edges through linear interpolation and extrapolation
        for (i = 1; i < nlay; i++)
        {
            Kitzmann::linear_log_interp(pe[i], pl[i - 1], pl[i], Tl[i - 1], Tl[i], Te__df_e[i]);
        }
        Te__df_e[0] = Tl[0] + (pe[0] - pe[1]) / (pl[0] - pe[1]) * (Tl[0] - Te__df_e[1]);
        Te__df_e[nlay1 - 1] = Tl[nlay - 1] + (pe[nlay1 - 1] - pe[nlay - 1]) / (pl[nlay - 1] - pe[nlay - 1]) *
            (Tl[nlay - 1] - Te__df_e[nlay - 1]);

        // Shortwave fluxes
        for (i = 0; i < nlay1; i++)
        {
            sw_down__df_e[i] = 0.0;
            sw_up__df_e[i] = 0.0;
        }
        for (int channel = 0; channel < 3; channel++)
        {
            // Find the opacity structure
            tau_struct(nlay, grav, pe, k_V_l, channel, tau_Ve__df_e);

            // Incident flux in band
            Finc_B = Finc * Beta_V[channel];

            // Calculate sw flux
            sw_grey_down(nlay, Finc_B, tau_Ve__df_e, sw_down_b__df_e, mu_s);

            // Sum all bands
            for (i = 0; i < nlay1; i++)
            {
                sw_down__df_e[i] = sw_down__df_e[i] + sw_down_b__df_e[i];
            }
        }

        // Long wave two-stream fluxes
        for (i = 0; i < nlay1; i++)
        {
            lw_down__df_e[i] = 0.0;
            lw_up__df_e[i] = 0.0;
        }
        for (int channel = 0; channel < 2; channel++)
        {
            // Find the opacity structure
            tau_struct(nlay, grav, pe, k_IR_l, channel, tau_IRe__df_e);

            // Blackbody fluxes (note divide by pi for correct units)
            for (i = 0; i < nlay1; i++)
            {
                be__df_e[i] = StBC * pow((double)(Te__df_e[i]), ((double)4.0)) / pi * Beta[channel];
            }

            // Calculate lw flux
            lw_grey_updown_linear(nlay, nlay1, be__df_e, tau_IRe__df_e, lw_up_b__df_e, lw_down_b__df_e,
                dtau__dff_l, del__dff_l, edel__dff_l, e0i__dff_l, e1i__dff_l,
                Am__dff_l, Bm__dff_l, lw_up_g__dff_l, lw_down_g__dff_l);
            //lw_grey_updown_poly(nlay, nlay1, be__df_e, tau_IRe__df_e, lw_up_b__df_e, lw_down_b__df_e,
            //dtau__dff_l, del__dff_l, edel__dff_l, e0i__dff_l, e1i__dff_l,
            //    e2i__dff_l, Am__dff_l, Bm__dff_l, Gm__dff_l, lw_up_g__dff_l, lw_down_g__dff_l);


            // Sum all bands
            for (i = 0; i < nlay1; i++)
            {
                lw_up__df_e[i] = lw_up__df_e[i] + lw_up_b__df_e[i];
                lw_down__df_e[i] = lw_down__df_e[i] + lw_down_b__df_e[i];
            }

        }

        // Net fluxes
        for (i = 0; i < nlay1; i++)
        {
            lw_net__df_e[i] = lw_up__df_e[i] - lw_down__df_e[i];
            sw_net__df_e[i] = sw_up__df_e[i] - sw_down__df_e[i];
            net_F[i] = lw_net__df_e[i] + sw_net__df_e[i];
        }

        net_F[nlay1 - 1] = Fint;



    }
