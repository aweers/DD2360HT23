#include <stdio.h>
#include <sys/time.h>

#define TPB 128
#define DataType double
#define DOUBLE_MIN -5
#define DOUBLE_MAX 5
#define NSTREAMS 4
#define S_SEG 33554432

#define CUDA_CHECK(call)                                          \
  if ((call) != cudaSuccess)                                      \
  {                                                               \
    fprintf(stderr, "CUDA error at %s %d\n", __FILE__, __LINE__); \
    return EXIT_FAILURE;                                          \
  }

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len)
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

  struct timeval computeTimer;
  double computeTime;

  //@@ Insert code below to read in inputLength from args
  if(argc > 1){
    inputLength = atoi(argv[1]);
  }

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  CUDA_CHECK(cudaHostAlloc((void **)&hostInput1, inputLength * sizeof(DataType), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void **)&hostInput2, inputLength * sizeof(DataType), cudaHostAllocDefault));
  CUDA_CHECK(cudaHostAlloc((void **)&hostOutput, inputLength * sizeof(DataType), cudaHostAllocDefault));
  resultRef = (DataType *)malloc(inputLength * sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  for(int i = 0; i < inputLength; ++i) {
    hostInput1[i] = randDouble(DOUBLE_MIN, DOUBLE_MAX);
    hostInput2[i] = randDouble(DOUBLE_MIN, DOUBLE_MAX);
    //printf("host1: %d", hostInput1[i]);
    resultRef[i] = hostInput1[i] + hostInput2[i];
  }

  int n_seg = (inputLength + S_SEG - 1) / S_SEG;
  printf("Number of segments: %d\n", n_seg);

  //@@ Insert code below to allocate GPU memory here
  CUDA_CHECK(cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType)));
  CUDA_CHECK(cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType)));
  CUDA_CHECK(cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType)));

  cudaStream_t streams[NSTREAMS];
  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  printf("Start of GPU computation\n");
  timerStart(&computeTimer);
  //@@ Insert code below to copy memory to the GPU, start kernel, and copy back for each stream
  for (int i = 0; i < n_seg; i++) {
    int offset = i * S_SEG;
    int size = (i == n_seg - 1) ? (inputLength - offset) : S_SEG;
    cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, streams[i % NSTREAMS]);
    cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, size * sizeof(DataType), cudaMemcpyHostToDevice, streams[i % NSTREAMS]);

    dim3 DimGrid((size + TPB - 1) / TPB, 1, 1);
    dim3 DimBlock(TPB, 1, 1);
    vecAdd<<<DimGrid, DimBlock, 0, streams[i % NSTREAMS]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, size);
  }

  //@@ Insert code below to copy memory to the GPU, start kernel, and copy back for each stream
  for (int i = 0; i < n_seg; i++) {
    int offset = i * S_SEG;
    int size = (i == n_seg - 1) ? (inputLength - offset) : S_SEG;
    
    cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, size * sizeof(DataType), cudaMemcpyDeviceToHost, streams[i % NSTREAMS]);
  }
  
  //@@ Insert code below to synchronize and destroy streams
  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
  computeTime = timerStop(&computeTimer);
  printf("End of GPU computation\n");

  //@@ Insert code below to compare the output with the reference
  double diff = 0.0;
  for(int i = 0; i < inputLength; ++i) {
    diff += abs(hostOutput[i] - resultRef[i]);
  }
  printf("Average difference: %f\n\n", diff/(double)inputLength);

  printf("Total GPU Time: %f ms\n", computeTime);


  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  free(resultRef);

  return 0;
}