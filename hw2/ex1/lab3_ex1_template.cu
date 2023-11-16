#include <stdio.h>
#include <sys/time.h>

#define TPB 128
#define DataType double
#define DOUBLE_MIN -5
#define DOUBLE_MAX 5

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = in1[i] + in2[i];
}

//@@ Insert code to implement timer start
void timerStart(struct timeval *start) {
  gettimeofday(start, NULL);
}

//@@ Insert code to implement timer stop
double timerStop(struct timeval *start) {
  struct timeval end;
  gettimeofday(&end, NULL);
  double time = (end.tv_sec - start->tv_sec) * 1000.0;
  time += (end.tv_usec - start->tv_usec) / 1000.0;
  return time;
}

double randDouble(double min, double max) {
  double scale = rand() / (double)RAND_MAX;
  return min + scale * (max-min);
}


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  struct timeval copyToDevice, copyFromDevice, kernelExecution;
  double copyToDeviceTime, copyFromDeviceTime, kernelExecutionTime;

  //@@ Insert code below to read in inputLength from args
  if(argc > 1){
    inputLength = atoi(argv[1]);
  }
  

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
  resultRef = (DataType *)malloc(inputLength * sizeof(DataType));
  
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for(int i = 0; i < inputLength; ++i) {
    hostInput1[i] = randDouble(DOUBLE_MIN, DOUBLE_MAX);
    hostInput2[i] = randDouble(DOUBLE_MIN, DOUBLE_MAX);
    //printf("host1: %d", hostInput1[i]);
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));


  //@@ Insert code to below to Copy memory to the GPU here
  timerStart(&copyToDevice);
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  copyToDeviceTime = timerStop(&copyToDevice);


  //@@ Initialize the 1D grid and block dimensions here
  dim3 DimGrid((inputLength+TPB-1)/TPB, 1, 1);
  dim3 DimBlock(TPB, 1, 1);


  //@@ Launch the GPU Kernel here
  timerStart(&kernelExecution);
  vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();
  kernelExecutionTime = timerStop(&kernelExecution);


  //@@ Copy the GPU memory back to the CPU here
  timerStart(&copyFromDevice);
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
  copyFromDeviceTime = timerStop(&copyFromDevice);


  //@@ Insert code below to compare the output with the reference
  double diff = 0.0;
  for(int i = 0; i < inputLength; ++i) {
    diff += abs(hostOutput[i] - resultRef[i]);
  }
  printf("Average difference: %f\n\n", diff/(double)inputLength);

  printf("Copy to Device Time: %f ms\n", copyToDeviceTime);
  printf("Kernel Execution Time: %f ms\n", kernelExecutionTime);
  printf("Copy from Device Time: %f ms\n", copyFromDeviceTime);


  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}