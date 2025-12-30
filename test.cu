#include <stdio.h> 
__global__ void hello() { } 
int main() { hello<<<1,1>>>(); printf("CUDA OK\n"); return 0; } 
