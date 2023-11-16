
#include <stdio.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop


int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU


  //@@ Insert code below to allocate GPU memory here


  //@@ Insert code to below to Copy memory to the GPU here


  //@@ Initialize the 1D grid and block dimensions here


  //@@ Launch the GPU Kernel here


  //@@ Copy the GPU memory back to the CPU here


  //@@ Insert code below to compare the output with the reference


  //@@ Free the GPU memory here

  //@@ Free the CPU memory here

  return 0;
}
