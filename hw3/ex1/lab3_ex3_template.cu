%%writefile lab2_ex1.cu
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS 4096
#define TPB 128
#define STRIPE 32

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
  unsigned int num_elements,
  unsigned int num_bins) {
  const int i = blockIdx.x * blockDim.x * STRIPE + threadIdx.x;
  __shared__ unsigned int private_histo[NUM_BINS];

  if(threadIdx.x == 0) {
    for(int j = 0; j < NUM_BINS; ++j) {
      private_histo[j] = 0;
    }
  }
  __syncthreads();

  for(int j = 0; j < STRIPE; ++j) {
    int index = i + j * blockDim.x;
    if(index < num_elements) {
      atomicAdd(&(private_histo[input[index]]), 1);
    }
  }
  __syncthreads();

  // Merge private histograms into global histogram
  if(threadIdx.x == 0) {
    for(int j = 0; j < NUM_BINS; ++j) {
      atomicAdd(&(bins[j]), private_histo[j]);
    }
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
  //@@ Insert code below to clean up bins that saturate at 127
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < num_bins)
    atomicMin(&bins[i], 127);
}


int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if(argc > 1){
    inputLength = atoi(argv[1]);
  }

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*) malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));

  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  for(int i = 0; i < inputLength; i++){
    hostInput[i] = rand() % NUM_BINS;
  }


  //@@ Insert code below to create reference result in CPU
  for(int i = 0; i < NUM_BINS; i++){
    resultRef[i] = 0;
  }
  for(int i = 0; i < inputLength; i++) {
    resultRef[hostInput[i]]++;
  }
  for(int i = 0; i < NUM_BINS; i++) {
    if(resultRef[i] > 127)
        resultRef[i] = 127;
  }


  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void**)&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void**)&deviceBins, NUM_BINS * sizeof(unsigned int));


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);


  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));


  // Pass a stripe to each kernel
  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((inputLength + STRIPE*TPB - 1) / (STRIPE*TPB));
  dim3 DimBlock(TPB);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<DimGrid, DimBlock>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  dim3 DimGrid2((NUM_BINS+TPB-1)/TPB, 1, 1);
  dim3 DimBlock2(TPB, 1, 1);


  //@@ Launch the second GPU Kernel here
  convert_kernel<<<DimGrid2, DimBlock2>>>(deviceBins, NUM_BINS);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  for(int i = 0; i < NUM_BINS; ++i) {
    if(hostBins[i] != resultRef[i]) {
      printf("Mismatch at index %d, host is %d, ref is %d\n", i, hostBins[i], resultRef[i]);
      break;
    }
  }

  if(inputLength < 100) {
    printf("Input:\n");
    for(int i = 0; i < inputLength; ++i) {
      printf("%d ", hostInput[i]);
    }
    printf("\n");

    // print differences
    for(int i = 0; i < NUM_BINS; ++i) {
      if(hostBins[i] != resultRef[i]) {
        printf("Mismatch at index %d, host is %d, ref is %d\n", i, hostBins[i], resultRef[i]);
      }
    }
  }


  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);


  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}