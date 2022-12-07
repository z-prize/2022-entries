#include "asm_fft_cuda.h"
#include "blst_ops.h"

#include "stdio.h"

__global__ void test2(void) {
	blst_fr two; 
	blst_fr_add(two, BLS12_377_ONE, BLS12_377_ONE);
	
	printf("two is %lld \n", two);
}

int main(void) {

	test2 <<<1,10>>>();
	cudaDeviceReset();
	
	return 0;	
}
