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

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

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
    int n_step[ncol] = { (double)(1e5) };

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






    /*
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

    */









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

    // print time difference <<<<<<<
    seconds = difftime(timer1, timer2);
    cout <<  n_step << " | " << "took: " << seconds << endl;

    */


    
    /// ///////////////////////////////////////////////////////////////////////////////////////////
    /// ///////////////////////////////////////////////////////////////////////////////////////////////
    
    const int arraySize = 5;
    const int aa[arraySize] = { 1, 2, 3, 4, 5 };
    const int bb[arraySize] = { 10, 20, 30, 40, 50 };
    int cc[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(cc, aa, bb, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        cc[0], cc[1], cc[2], cc[3], cc[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
