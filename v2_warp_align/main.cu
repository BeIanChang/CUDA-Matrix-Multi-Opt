#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define N 1024
#define ITERATIONS 10
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32
#define BLOCK_SIZE 32
using namespace std;


__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if( (i<n) && (j<n) )
    {
        float tmp = 0;
        for(int k=0; k<n; k++)
        {
            tmp += A[i*n+k]*B[k*n+j];
        }
        C[i*n+j]=a*tmp + b * C[i*n+j];
    }
}



void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else 
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else 
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    if(i<10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>0.0005)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(){
  float *A, *B, *C_cpu, *C_gpu_final;
  A = (float *)malloc(sizeof(float) * N * N);
  B = (float *)malloc(sizeof(float) * N * N);
  C_cpu = (float *)malloc(sizeof(float) * N * N);
  C_gpu_final = (float *)malloc(sizeof(float) * N * N);
  //float A[N][N], B[N][N], C_cpu[N][N], C_gpu_final[N][N];
  float a=0.5, b=0.3;
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      A[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      B[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      C_cpu[i*N+j]=0;
      C_gpu_final[i*N+j]=0;
    }
  }
  // CPU straight perform for validation  
    for(int j=0; j<N; j++)
    {
        for(int i=0; i<N; i++)
        {
            C_cpu[i*N+j]+=b*C_cpu[i*N+j];
            for(int k=0; k<N; k++){
                C_cpu[i*N+j] += a*A[i*N+k]*B[k*N+j];
            }
        }
    }

  float *A_gpu;
  float *B_gpu;
  float *C_gpu;
  cudaMalloc((void **)&A_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&B_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&C_gpu, sizeof(float)*N*N);
  cudaMemcpy(A_gpu, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((size_t)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_X) ), 
            (size_t)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_Y)) );

  sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
  cudaDeviceSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
  compare(C_cpu, C_gpu_final, N*N);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){

    sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);

  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double time = milliseconds / 1000 / ITERATIONS;

  
  double flops = 2.0*N*N*N;
  double gflopsPerSecond = flops/(1000000000)/time;
  double GB = (double)(N)*N*4/1000000000;
  double GBpS = (double)(N)*N*4/1000000000/time;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",GBpS);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("GB=%lf\n",GB);
  printf("time(s)=%lf\n",time);

  
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  return 0;
}
