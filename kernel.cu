#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include "IC_mod.h"


using namespace std;

cudaError_t addWithCuda(
    double* T,
    double* pl,
    double* pe,
    double* met,
    double* gam_V,
    double* gam_1,
    double* gam_2,
    double* Beta_V,
    double* Beta,
    double* net_F,
    double* mu_s,
    double* F0,
    double* Fint,
    double* grav,
    double* AB,
    double* cp_air,
    double* kappa_air,
    double* t_step,
    int* n_step,
    double* host_k_IR_l,
    double* host_k_V_l,
    double* dT_rad,
    double* dT_conv,
    unsigned int nlay,
    unsigned int ncol
);



int main()
{
    

    const double StBC = 5.670374419e-8;


    // Number of columns, layers and edges
    const int ncol = 2;
    const int nlay = 52;
    const int nlay1 = nlay + 1;



    const std::string frmat = "(I6.6)";

    // Read in sigma hybrid grid values
    const std::string a_sh = "sig_hyb_HJ_53_a.txt";
    const std::string b_sh = "sig_hyb_HJ_53_b.txt";

    // Surface pressure (pa)
    const double p0 = 2.2e7;

    // step in seconds
    double t_step[ncol] = { 30.0 };

    // Number of steps
    double n_step[ncol] = { 1e5 };

    // Constants
    //double R = 8.31446261815324; //     ! Ideal gas constant
    double Rd_air[ncol]; //          ! Specific gas constant
    double cp_air[ncol]; //           ! Rd_air/kappa_air ! Heat capacity of air
    double kappa_air[ncol];// ! kappa = Rd/cp
    double grav[ncol]; //              ! Gravity
    double met[ncol];   //                ! Metallicity in dex solar, solar[M/H] = 0.0


    //! short wave Zenith angle
    double mu_s[ncol];

    double Tirr[ncol]; //               ! Irradiation temperature
    double Tint[ncol]; //                 ! Internal temperature

    double F0[ncol];   //        ! Substellar point irradiation flux

    double Fint[ncol];//      ! Internal flux

    double k_IR[ncol];        //           ! Constant IR opacity
    double k_V[ncol];//! Constant V opacity

    double gam[ncol]; //               ! Gamma ratio

    // Parmentier IC and parameters
    int iIC = 4; //       ! IC choice here
    bool corr = true; //  ! Do adibatic correction

    int table_num = 1; // ! Table 1 = with TiO/VO, Table 2 = without TiO/VO

    for (size_t i = 0; i < ncol; i++)
    {
        Rd_air[i] = 3556.8;
        cp_air[i] = 1.3e4;
        kappa_air[i] = Rd_air[i] / cp_air[i];
        grav[i] = 10.0;
        met[i] = 0.0;
        mu_s[i] = 1.0 / sqrt(3.0);
        Tirr[i] = 1000.0;
        Tint[i] = 500.0;
        F0[i] = StBC * pow(Tirr[i], 4);
        Fint[i] = StBC * pow(Tint[i], 4);
        k_IR[i] = 1e-3;
        k_V[i] = 6e-4 * sqrt(Tirr[i] / 2000.0);
        gam[i] = k_V[i] / k_IR[i];
    }


    //std::fstream myfile("sig_hyb_HJ_53_a.txt", std::ios_base::in);

    //FILE *file;
    //file = fopen("sig_hyb_HJ_53_a.txt", "r");
    //std::ifstream input( "sig_hyb_HJ_53_a.txt" );


    //int x = 0;
    int i = 0;
    double a[ncol * nlay1];
    double b[ncol * nlay1];
    double pe[ncol * nlay1];
    double pu[ncol];
    double pl[ncol * nlay];
    double k_V_l[3 * ncol * nlay];
    double k_IR_l[2 * ncol * nlay];
    double k_V_l_1D[ncol * nlay];
    double k_IR_l_1D[ncol * nlay];
    double T[ncol * nlay];
    double prc[ncol];
    double Teff[ncol];
    double AB[ncol];
    double gam_V[ncol * 3];
    double Beta_V[ncol * 3];
    double Beta[ncol * 2];
    double gam_1[ncol];
    double gam_2[ncol];
    double gam_P[ncol];
    double tau_lim[ncol];

    //double t_tot = 0.0;
    //int inan = 0;
    //int k=0;
    //int n=0;
    double dT_rad[ncol * nlay];
    double dT_conv[ncol * nlay];
    double net_F[ncol * nlay1];

    //double seconds=0;
    time_t timer1, timer2;

    std::ofstream myfile;

    // Read in sigma hybrid grid values
    std::ifstream inFile;
    inFile.open(a_sh);
    if (!inFile)
    {
        //cout << "\nError opening the file: " << a_sh << endl;
        return 13;
    }
    for (i = 0; i < nlay1; i++)
    {
        inFile >> a[i];
    }
    inFile.close();

    inFile.open(b_sh);
    if (!inFile)
    {
        //cout << "\nError opening the file: " << b_sh << endl;
        return 13;
    }
    for (i = 0; i < nlay1; i++)
    {
        inFile >> b[i];
    }
    inFile.close();

    cout.precision(17);


    // Contruct pressure array in pa

    for (int c = 0; c < ncol; c++)
    {
        for (i = 0; i < nlay1; i++)
        {
            pe[c * nlay1 + i] = a[c * c + i] + b[c * nlay1 + i] * p0;
           // cout << "pe[i]  " << pe[c * nlay1 + i] << endl;

        }

        pu[c] = pe[c * nlay1 + 0];

    }





    //! Pressure layers

    for (int c = 0; c < ncol; c++)
    {
        for (i = 0; i < nlay; i++)
        {
            pl[c * nlay + i] = (pe[c * nlay + i + 1] - pe[c * nlay + i]) / (logl(pe[c * nlay + i + 1]) - logl(pe[c * nlay + i]));

        }
    }



    /*
    cout << "Tint | Tirr | p0 | pu | mu_s | grav " << endl;
    cout << Tint[0] << " | " << Tirr[0] << " | " <<
        p0 / 1e5 << " | " << pu[0] / 1e5 << " | " <<
        mu_s[0] << " | " << grav[0] << endl;
    cout << "-------------------------------" << endl;
    */

    // Semi-grey atmosphere values (here they are not used, but just need to be passed to IC routine)

    for (int c = 0; c < ncol; c++)
    {
        for (i = 0; i < nlay; i++)
        {
            k_V_l[c * nlay + 0 + i] = k_V[c];
            k_IR_l[c * nlay + 0 + i] = k_IR[c];
            k_V_l_1D[c * nlay + i] = k_V[c];
            k_IR_l_1D[c * nlay + 0 + i] = k_IR[c];
        }
    }

    double fl = (double)1.0;

    double tau_hf_e[nlay1] = { 0 };
    double kRoss_hf_e[nlay1];
    double tau_IRl_hf_l[0 + nlay];
    double gradrad_hf_l[0 + nlay] = { 0.0 }, gradad_hf_l[0 + nlay] = { 0.0 };


    double work_pl[nlay];
    double work_pe[nlay1];
    double work_T[nlay];

    double work_gam_V[3];
    double work_Beta_V[3];
    double work_Beta[2];



    for (int c = 0; c < ncol; c++)
    {
        for (int i = 0; i < nlay; i++)
        {
            work_pl[i] = pl[c * nlay + i];
            work_T[i] = T[c * nlay + i];
        }
        for (int i = 0; i < nlay1; i++)
        {
            work_pe[i] = pe[c * nlay1 + i];
        }

        //  Parmentier IC 
        IC_profile(iIC, corr, nlay,
            p0, work_pl, work_pe, k_V_l_1D, k_IR_l_1D, Tint[c],
            mu_s[c], Tirr[c], grav[c], fl,
            work_T, prc[c], table_num, met[c], tau_hf_e, kRoss_hf_e, tau_IRl_hf_l, gradrad_hf_l, gradad_hf_l);

        // Parmentier opacity profile parameters - first get Bond albedo
        Teff[c] = powl((powl(Tint[c], 4) + (1.0 / sqrtl((double)3.0)) *
            powl(Tirr[c], 4)), 0.25);
        Bond_Parmentier(Teff[c], grav[c], AB[c]);

        // Recalculate Teff and then find parameters
        Teff[c] = powl((powl(Tint[c], 4) + (((double)1.0) - AB[c]) * mu_s[c] *
            powl(Tirr[c], 4)), (0.25));

        gam_Parmentier(Teff[c], table_num, work_gam_V,
            work_Beta_V, work_Beta, gam_1[c], gam_2[c], gam_P[c], tau_lim[c]);

        for (int i = 0; i < 3; i++)
        {
            gam_V[c * 3 + i] = work_gam_V[i];
            Beta_V[c * 3 + i] = work_Beta_V[i];
        }
        for (int i = 0; i < 2; i++)
        {
            Beta[c * 2 + i] = work_Beta[i];
        }
    }


    /*
    // Print variables from Parmentier non-grey scheme
    cout << "Teff | AB | gam_V [0,1,2] | Beta_V [0,1,2] | Beta[0,1] | gam_1 | gam_2 | gam_P | tau_lim | prc" << endl;
    cout << Teff[0] << " | " << AB[0] << " | " <<
        gam_V[0] << " | " << Beta_V[0] << " | " <<
        Beta[0] << " | " << gam_1 << " | " <<
        gam_2 << " | " << gam_P << " | " <<
        tau_lim << " | " << prc[0] / 1e5 << endl;
    cout << "   " << " | " << "   " << " | " <<
        gam_V[1] << " | " << Beta_V[1] << " | " <<
        Beta[1] << " | " << "   " << " | " <<
        "   " << " | " << "   " << " | " <<
        "   " << " | " << "   " << endl;
    cout << "   " << " | " << "   " << " | " <<
        gam_V[2] << " | " << Beta_V[2] << " | " <<
        "   " << " | " << "   " << " | " <<
        "   " << " | " << "   " << " | " <<
        "   " << " | " << "   " << endl;
    cout << "-------------------------------" << endl;
    */

    // Print T-p profile
    // Write out initial conditions
    myfile.open("FMS_RC_ic.out");

    for (i = 0; i < nlay; i++)
    {
        //cout << i << " | " << pl[i] / 1e5 << " | " << T[i] << endl;
        myfile << i << " " << pl[i] / 1e5 << " " << T[i] << endl;
    }
    myfile.close();


    // Time stepping loop
    //cout << "Start timestepping" << endl;
    ////  time code here
    time(&timer2);





    //////////////////////////////////////////////////////////////////////////

    // Initialize device parameters
    /*
    size_t double3;
    double3 = 3 * sizeof(float);
    double *host_k_IR_l, *host_k_V_l;
    host_k_IR_l = (double*)malloc(double3);
    host_k_V_l = (double*)malloc(double3);


    for (int i = 0; i < nlay; i++)
    {
        for (int k = 0; k < 2; k++)
        {
            host_k_IR_l[2 * nlay] = k_IR_l[k][i];
        }
    }

    for (int i = 0; i < nlay; i++)
    {
        for (int k = 0; k < 3; k++)
        {
            host_k_V_l[3 * nlay] = k_V_l[k][i];
        }
    }

    */






    
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda( T,
        pl,
        pe,
        met,
        gam_V,
        gam_1,
        gam_2,
        Beta_V,
        Beta,
        net_F,
        mu_s,
        F0,
        Fint,
        grav,
        AB,
        cp_air,
        kappa_air,
        t_step,
        n_step,
        k_IR_l,
        k_V_l,
        dT_rad,
        dT_conv,
        nlay,
        ncol);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    








    ///////////////////////////////////////////////////////////////////////////////



        // cpu time end
    time(&timer1);




    // Output

    /*
    cout << "sec: | " << "hours: | " << "days: | " << endl;

    cout << t_tot << " | " << t_tot / 60.0 / 60.0 << " | " <<
        t_tot / 60.0 / 60.0 / 24.0 << endl;

    */
    /*
    // !write (iname,frmat) int(t_tot/60.0_dp/60.0_dp/24.0_dp)
    //open(newunit=u,file='FMS_RC_pp.out',action='readwrite')
    myfile.open("FMS_RC_pp.out");

    for (i = 0; i <  nlay; i++)
    {

        myfile << i << " | " << pl[i] << " | " << T[i] << " | " <<
            dT_rad[i] << " | " << dT_conv[i] << " | " << k_V_l[0][i] <<
            " | " << k_V_l[1][i] << " | " << k_V_l[2][i] << " | " <<
            k_IR_l[0][i] << " | " << k_IR_l[1][i] << endl;
    }
    myfile.close();


    cout <<  Tint << " | " <<  Tirr << " | " <<
         p0 << " | " << pu << " | " <<  mu_s << " | " <<
         gam << endl;

    */
    // print time difference <<<<<<<
    double seconds;
    seconds = difftime(timer1, timer2);
    cout <<  n_step << " | " << "took: " << seconds << endl;

    


    
    /// ///////////////////////////////////////////////////////////////////////////////////////////
    /// ///////////////////////////////////////////////////////////////////////////////////////////////
    

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.


cudaError_t addWithCuda(
    double* T,
    double* pl,
    double* pe,
    double* met,
    double* gam_V,
    double* gam_1,
    double* gam_2,
    double* Beta_V,
    double* Beta,
    double* net_F,
    double* mu_s,
    double* F0,
    double* Fint,
    double* grav,
    double* AB,
    double* cp_air,
    double* kappa_air,
    double* t_step,
    int* n_step,
    double* host_k_IR_l,
    double* host_k_V_l,
    double* dT_rad,
    double* dT_conv,
    unsigned int nlay,
    unsigned int ncol) {

    cudaError_t cudaStatus;



    // Initialize device parameters

    unsigned int* dev_nlay;
    unsigned int* dev_ncol;
    //const int *dev_nlay1 ;
    double* dev_dT_rad;
    double* dev_dT_conv;
    double* dev_T;            /// parallel parameter
    double* dev_pl;
    double* dev_pe;
    double* dev_met;
    double* dev_k_IR_l;
    double* dev_k_V_l;
    double* dev_gam_V;
    double* dev_gam_1;
    double* dev_gam_2;
    double* dev_Beta_V;
    double* dev_Beta;
    double* dev_net_F;  /// parallel parameter
    double* dev_mu_s;
    double* dev_F0;
    double* dev_Fint;
    double* dev_grav;
    double* dev_AB;
    double* dev_cp_air;
    double* dev_kappa_air;
    double* dev_t_step;
    int* dev_n_step;
    //int* dev_num;




    //Kitzman working variables nlay1
    double* tau_Ve__df_e, * tau_IRe__df_e, * Te__df_e, * be__df_e,
        * sw_down__df_e, * sw_down_b__df_e, * sw_up__df_e,
        * lw_down__df_e, * lw_down_b__df_e,
        * lw_up__df_e, * lw_up_b__df_e,
        * lw_net__df_e, * sw_net__df_e,

        // lw_grey_updown_linear working variables nlay
        * dtau__dff_l, * del__dff_l,
        * edel__dff_l, * e0i__dff_l, * e1i__dff_l,
        * Am__dff_l, * Bm__dff_l,
        * lw_up_g__dff_l, * lw_down_g__dff_l,

        // dry_adj_Ray working variables nlay
        * Tl_cc__df_l, * d_p__df_l;




    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_nlay, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_ncol, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_dT_rad, ncol * nlay * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_dT_conv, (ncol * nlay * sizeof(double)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_T, (ncol * nlay * sizeof(double)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pl, (ncol * nlay * sizeof(double)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pe, (ncol * (1 + nlay) * sizeof(double)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_dT_rad, ncol * (nlay * sizeof(double)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_met, (ncol * sizeof(double)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_k_IR_l, 2 * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_k_V_l, 3 * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_gam_V, ncol * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_gam_1, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_gam_2, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_Beta_V, ncol * 3 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_Beta, ncol * 2 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_net_F, ncol * (1 + nlay) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_mu_s, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_F0, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_Fint, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_grav, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_AB, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_cp_air, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_kappa_air, ncol * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_t_step, ncol * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_n_step, ncol * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    //Kitzman working variables
    cudaStatus = cudaMalloc((void**)&tau_Ve__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&tau_IRe__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&Te__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&be__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&sw_down__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&sw_down_b__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&sw_up__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&lw_down__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&lw_down_b__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&lw_up__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&lw_up_b__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&lw_net__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&sw_net__df_e, ncol * (nlay + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // lw_grey_updown_linear working variables
    cudaStatus = cudaMalloc((void**)&dtau__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&del__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&edel__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&e0i__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&e1i__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&Am__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&Bm__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&lw_up_g__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&lw_down_g__dff_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // dry_adj_Ray
    cudaStatus = cudaMalloc((void**)&Tl_cc__df_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_p__df_l, ncol * nlay * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    // Allocate memory on the device

    //cudaMalloc(&dev_nlay, sizeof(int));
    //cudaMalloc(&dev_nlay1, sizeof(const int));
    //cudaMalloc((void**)&dev_dT_rad, nlay * sizeof(double));
    //cudaMalloc((void**)&dev_dT_conv, nlay * sizeof(double));
    //cudaMalloc((void**)&dev_T, nlay * sizeof(double));
    //cudaMalloc((void**)&dev_pl, nlay * sizeof(double));
    //cudaMalloc((void**)&dev_pe, nlay1 * sizeof(double));
    //cudaMalloc(&dev_met, sizeof(double));
    //cudaMalloc((void**)&dev_k_IR_l, 2 * nlay * sizeof(double));
    //cudaMalloc((void**)&dev_k_V_l, 3 * nlay * sizeof(double));
    //cudaMalloc((void**)&dev_gam_V, 3 * sizeof(double));
    //cudaMalloc((void**)&dev_gam_1, sizeof(double));
    //cudaMalloc((void**)&dev_gam_2, sizeof(double));
    //cudaMalloc((void**)&dev_Beta_V, 3 * sizeof(double));
    //cudaMalloc((void**)&dev_Beta, 2 * sizeof(double));
    //cudaMalloc((void**)&dev_net_F, (1+nlay) * sizeof(double));
    //cudaMalloc((void**)&dev_mu_s, sizeof(double));
    //cudaMalloc((void**)&dev_F0, sizeof(double));
    //cudaMalloc((void**)&dev_Fint, sizeof(double));
    //cudaMalloc((void**)&dev_grav, sizeof(double));
    //cudaMalloc((void**)&dev_AB, sizeof(double));
    //cudaMalloc((void**)&dev_cp_air, sizeof(double));
    //cudaMalloc((void**)&dev_kappa_air, sizeof(double));
    //cudaMalloc((void**)&dev_t_step, sizeof(int));
    //cudaMalloc((void**)&dev_n_step, sizeof(int));
    //cudaMalloc((void**)&dev_num, sizeof(int));




    // Copy data from the host to the device (CPU -> GPU)

    cudaStatus = cudaMemcpy(dev_nlay, &nlay, sizeof(const int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    //cudaMemcpy(dev_nlay1,  nlay,  sizeof(const int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_dT_rad,  nlay,  nlay * sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_dT_conv,  nlay,  nlay * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_T, T, nlay * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_pl, pl, nlay * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_pe, pe, (nlay + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_met, met, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_k_IR_l, host_k_IR_l, 2 * nlay * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_k_V_l, host_k_V_l, 3 * nlay * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_gam_V, gam_V, 3 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_gam_1, gam_1, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_gam_2, gam_2, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_Beta_V, Beta_V, 3 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_Beta, Beta, 2 * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_net_F, net_F, (nlay + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_mu_s, mu_s, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_F0, F0, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_Fint, Fint, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_grav, grav, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_AB, AB, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_cp_air, cp_air, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_kappa_air, kappa_air, sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_t_step, t_step, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_n_step, n_step, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }






    // Threads per CTA (1024)
    //dim3 NUM_THREADS = 256;   //1 << 10;
    //dim3 NUM_THREADS = 2;
    //const int NUM_THREADS = 2;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    //dim3 NUM_BLOCKS = 1;   // (N + NUM_THREADS - 1) / NUM_THREADS;

    //dim3 NBRT((2 / NUM_THREADS) + 1, 1, 1);
    //dim3 NB = 2;


    // Launch the kernel on the GPU
    /*
    kernel_RT_loop << <NBRT, NUM_THREADS >> > (
        dev_dT_rad,
        dev_dT_conv,
        dev_T,
        dev_pl,
        dev_pe,
        dev_met,
        dev_k_IR_l,
        dev_k_V_l,
        dev_gam_V,
        dev_gam_1,
        dev_gam_2,
        dev_Beta_V,
        dev_Beta,
        dev_net_F,
        dev_mu_s,
        dev_F0,
        dev_Fint,
        dev_grav,
        dev_AB,
        dev_cp_air,
        dev_kappa_air,
        dev_t_step,
        dev_n_step,
        0,
        nlay,

        tau_Ve__df_e, tau_IRe__df_e, Te__df_e, be__df_e, //Kitzman working variables
        sw_down__df_e, sw_down_b__df_e, sw_up__df_e,
        lw_down__df_e, lw_down_b__df_e,
        lw_up__df_e, lw_up_b__df_e,
        lw_net__df_e, sw_net__df_e,

        dtau__dff_l, del__dff_l, // lw_grey_updown_linear working variables
        edel__dff_l, e0i__dff_l, e1i__dff_l,
        Am__dff_l, Bm__dff_l,
        lw_up_g__dff_l, lw_down_g__dff_l,

        Tl_cc__df_l, d_p__df_l); // dry_adj_Ray working variables
        */


                // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error;
            }

            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
            }







            // Copy output vector from GPU buffer to host memory.

            cudaStatus = cudaMemcpy(dT_rad, dev_dT_rad, nlay * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(dT_conv, dev_dT_conv, nlay * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(T, dev_T, nlay * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(host_k_IR_l, dev_k_IR_l, 2 * nlay * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(host_k_V_l, dev_k_V_l, 3 * nlay * sizeof(int), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error;
            }
            cudaStatus = cudaMemcpy(net_F, dev_net_F, (nlay + 1) * sizeof(double), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error;
            }


        Error:
            // Release GPU memory


            //cudaFree(dev_nlay);
            //cudaFree(dev_nlay1);
            cudaFree(dev_dT_rad);
            cudaFree(dev_dT_conv);
            cudaFree(dev_T);
            cudaFree(dev_pl);
            cudaFree(dev_pe);
            cudaFree(dev_met);
            cudaFree(dev_k_IR_l);
            cudaFree(dev_k_V_l);
            cudaFree(dev_gam_V);
            cudaFree(dev_gam_1);
            cudaFree(dev_gam_2);
            cudaFree(dev_Beta_V);
            cudaFree(dev_Beta);
            cudaFree(dev_net_F);
            cudaFree(dev_mu_s);
            cudaFree(dev_F0);
            cudaFree(dev_Fint);
            cudaFree(dev_grav);
            cudaFree(dev_AB);
            cudaFree(dev_cp_air);
            cudaFree(dev_kappa_air);
            cudaFree(dev_t_step);
            cudaFree(dev_n_step);
            //cudaFree(dev_num);

            cudaFree(tau_Ve__df_e);
            cudaFree(tau_IRe__df_e);
            cudaFree(Te__df_e);
            cudaFree(be__df_e);
            cudaFree(sw_down__df_e);
            cudaFree(sw_down_b__df_e);
            cudaFree(sw_up__df_e);
            cudaFree(lw_down__df_e);
            cudaFree(lw_down_b__df_e);
            cudaFree(lw_up__df_e);
            cudaFree(lw_up_b__df_e);
            cudaFree(lw_net__df_e);
            cudaFree(sw_net__df_e);
            cudaFree(dtau__dff_l);
            cudaFree(del__dff_l);
            cudaFree(edel__dff_l);
            cudaFree(e0i__dff_l);
            cudaFree(e1i__dff_l);
            cudaFree(Am__dff_l);
            cudaFree(Bm__dff_l);
            cudaFree(lw_up_g__dff_l);
            cudaFree(lw_down_g__dff_l);

            cudaFree(Tl_cc__df_l);
            cudaFree(d_p__df_l);

            return cudaStatus;

        }
