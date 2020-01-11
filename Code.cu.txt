// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cufft.h>
#include <cutil_inline.h>
#include <shrQATest.h>
void runTest(int argc, char** argv);


#define SIGNAL_SIZE 16

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
    runTest(argc, argv);
}


void runTest(int argc, char** argv) 
{

    printf("[1DCUFFT] is starting...\n");

    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice( cutGetMaxGflopsDeviceId() );
    cufftComplex* h_signal=(cufftComplex*)malloc(sizeof(cufftComplex) * SIGNAL_SIZE);
    // Allocate host memory for the signal
    //Complex* h_signal = (Complex*)malloc(sizeof(Complex) * SIGNAL_SIZE);
    // Initalize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE; ++i) {
        h_signal[i].x = rand() / (float)RAND_MAX;
        h_signal[i].y = 0;
    }




    int mem_size = sizeof(cufftComplex) * SIGNAL_SIZE;

    // Allocate device memory for signal
    cufftComplex* d_signal;
    cutilSafeCall(cudaMalloc((void**)&d_signal, mem_size));

    // Copy host memory to device
    cutilSafeCall(cudaMemcpy(d_signal, h_signal, mem_size,
                              cudaMemcpyHostToDevice));



    // CUFFT plan
    cufftHandle plan;
    cufftSafeCall(cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_C2C, 1));

    // Transform signal 
    printf("Transforming signal cufftExecC2C\n");
    cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));


    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));

    // Copy device memory to host
    cufftComplex* h_inverse_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * SIGNAL_SIZE);;
    cutilSafeCall(cudaMemcpy(h_inverse_signal, d_signal, mem_size,
                              cudaMemcpyDeviceToHost));

    for(int i=0;i< SIGNAL_SIZE;i++){
        h_inverse_signal[i].x= h_inverse_signal[i].x/(float)SIGNAL_SIZE;
        h_inverse_signal[i].y= h_inverse_signal[i].y/(float)SIGNAL_SIZE;

        printf("first : %f %f  after %f %f \n",h_signal[i].x,h_signal[i].y,h_inverse_signal[i].x,h_inverse_signal[i].y);
    }  



    //Destroy CUFFT context
    cufftSafeCall(cufftDestroy(plan));

    // cleanup memory
    free(h_signal);

    free(h_inverse_signal);
    cutilSafeCall(cudaFree(d_signal));
    cutilDeviceReset();
}