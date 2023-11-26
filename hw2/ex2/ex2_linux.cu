#include <stdio.h>
#include <sys/time.h>

#define TPB 16
#define DataType double
#define DOUBLE_MIN -5
#define DOUBLE_MAX 5

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s, line: %d\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                    int numAColumns, int numBRows, int numBColumns){
    //@@ Insert code to implement matrix multiplication here
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_y < numARows && index_x < numBColumns)
    {
        DataType sum = 0;
        for (int k = 0; k < numAColumns; k++){
            sum += A[index_y * numAColumns + k] * B[k * numBColumns + index_x]; 
        }
        C[index_y * numBColumns + index_x] = sum;   
    }
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
    DataType *hostA; // The A matrix
    DataType *hostB; // The B matrix
    DataType *hostC; // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    struct timeval copyToDevice, copyFromDevice, kernelExecution;
    double copyToDeviceTime, copyFromDeviceTime, kernelExecutionTime;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns from args
    if (argc >= 4)
    {
        numARows = atoi(argv[1]);
        numAColumns = atoi(argv[2]);
        numBRows = numAColumns;
        numBColumns = atoi(argv[3]);
        numCRows = numARows;
        numCColumns = numBColumns;
    }
    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    
    //@@ Insert code below to allocate Host memory for input and output
    hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
    hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
    hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    
    //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
    //printf("\n\nhostA:\n");
    for(int i = 0; i < numARows; i++){
        for(int j = 0; j < numAColumns; j++){
            hostA[i * numAColumns + j] = randDouble(DOUBLE_MIN, DOUBLE_MAX);    
            //printf("%.3f ", hostA[i * numAColumns + j]);
        }
        //printf("\n");
    }

    //printf("\n\nhostB:\n");
    for(int i = 0; i < numBRows; i++){
        for(int j = 0; j < numBColumns; j++){
            hostB[i * numBColumns + j] = randDouble(DOUBLE_MIN, DOUBLE_MAX);  
            //printf("%.3f ", hostB[i * numBColumns + j]);  
        }
        //printf("\n");
    }
    
    //printf("\n\nresultRef:\n");
    for(int i = 0; i < numCRows; i++){
        for(int j = 0; j < numCColumns; j++){
            resultRef[i * numCColumns + j] = 0; 
  
            for (int k = 0; k < numAColumns; k++) { 
                resultRef[i * numCColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j]; 
            } 
            //printf("%.3f ", resultRef[i * numCColumns + j]);  
        }
        //printf("\n");
    }
        

    //@@ Insert code below to allocate GPU memory here
    CUDA_CHECK(cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType)));
    CUDA_CHECK(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType)));
    CUDA_CHECK(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType)));


    //@@ Insert code to below to Copy memory to the GPU here
    timerStart(&copyToDevice);
    CUDA_CHECK(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice));
    copyToDeviceTime = timerStop(&copyToDevice);

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid((numCColumns + TPB - 1) / TPB, (numCRows + TPB - 1) / TPB, 1);
    dim3 DimBlock(TPB, TPB, 1);

    //@@ Launch the GPU Kernel here
    timerStart(&kernelExecution);
    gemm<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numCColumns);
    cudaDeviceSynchronize();
    kernelExecutionTime = timerStop(&kernelExecution);
    
    //@@ Copy the GPU memory back to the CPU here
    timerStart(&copyFromDevice);
    CUDA_CHECK(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost));
    copyFromDeviceTime = timerStop(&copyFromDevice);

    //@@ Insert code below to compare the output with the reference
    double diff = 0.0;
    //printf("\n\nhostC:\n");
    for(int i = 0; i < numCRows; i++){
        for(int j = 0; j < numCColumns; j++) {
            diff += abs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]);
            //printf("%.3f ", hostC[i * numCColumns + j]);  
        }
        //printf("\n");
    }
    
    printf("Average difference: %f\n\n", diff/(double)(numCRows * numCColumns));

    printf("\nCopy to Device Time: %f ms\n", copyToDeviceTime);
    printf("Kernel Execution Time: %f ms\n", kernelExecutionTime);
    printf("Copy from Device Time: %f ms\n", copyFromDeviceTime);

    //@@ Free the GPU memory here
    cudaFree((void *) deviceA);
    cudaFree((void *) deviceB);
    cudaFree((void *) deviceC);

    //@@ Free the CPU memory here
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

