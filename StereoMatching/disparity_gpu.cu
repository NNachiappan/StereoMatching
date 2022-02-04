#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>

#include <opencv2/core.hpp>

#include "disparity_gpu.h"

using namespace std;


//Compare two windows using SSD
__device__ unsigned int ssd(int leftIdx, int rightIdx, unsigned char* leftShared, unsigned char* rightShared, short windSize, int n) {

	unsigned int SSD = 0;

	for (int i = 0; i < windSize; i++) {
		for (int j = 0; j < windSize; j++) {

			int idx = j + (i * (n+windSize-1));
			int diff = leftShared[leftIdx + idx] - rightShared[rightIdx + idx];

			SSD += diff * diff;
		}
	}

	return SSD;
};


//Main kernel function
__global__ void disparity(unsigned short int* dispImage, unsigned char* left, unsigned char* right, int m, int n, short windSize, const int numCalcs, const int pixPBlock, unsigned int* disps) {


	//Shared memory declaration
	extern __shared__ unsigned char shared[];
	unsigned char *leftShared = shared;
	unsigned char *rightShared = &leftShared[pixPBlock];

	//load left and right image data into shared memory specific to each block
	for (int i = 0; i < ceilf((float)pixPBlock/blockDim.x); i++) {
		int idx = threadIdx.x + i * blockDim.x;
		if (idx < pixPBlock) {
			leftShared[idx] = left[idx + blockIdx.x * (n + windSize-1)];
			rightShared[idx] = right[idx + blockIdx.x * (n + windSize-1)];
		}
	}

	__syncthreads();

	//Threshold
	unsigned short int thresh = (int) (floorf(0.1 * n)+1);


	//Calculate all disparities within the threshold range.
	for (int i = 0; i < ceilf((float)numCalcs / blockDim.x); i++) {
		int idx = threadIdx.x + i * blockDim.x;
		
		if (idx < numCalcs) {
			//Get the index of the current pixel of the row in the left image and
			// the index of the pixel it is being compared to in the right image.
			int leftIdx = (int) floorf((float)idx/thresh);
			int rightIdx = idx - (leftIdx * thresh);

			int rightSSDIdx = fmaxf(rightIdx, leftIdx - (thresh - rightIdx));

			if (rightIdx <= leftIdx) {
				disps[(rightIdx*n) + leftIdx + numCalcs * blockIdx.x] = ssd(leftIdx, rightSSDIdx, leftShared, rightShared, windSize, n);
			}
		}
	}

	__syncthreads();


	//Find min ssd and copy disparity to global memory
	for (int i = 0; i < ceilf((float)n/blockDim.x); i++) {
		int idx = threadIdx.x + i * blockDim.x;

		if (idx < n) {
			unsigned int minSSD = 4294967295;
			unsigned short int minJ = 0;

			for (int j = 0; j < thresh; j++) {
				if (disps[idx + (j*n) + numCalcs * blockIdx.x] < minSSD) {
					minSSD = disps[idx + (j * n) + numCalcs * blockIdx.x];
					minJ = j;
				}
			}

			//copy disparity to global mem
			dispImage[idx + blockIdx.x * n] = (int) fminf(idx - minJ, thresh - minJ);
			
		}
	}

}

void disparityGPU(cv::Mat& dispImage, cv::Mat& left, cv::Mat& right, int m, int n, short windSize) {

	//Timer code
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);



	unsigned short int* d_disparity;
	unsigned char *d_left, *d_right;
	
	unsigned int* d_disps;
	

	const int numCalcs = n * (floor(n * 0.1)+1);
	const int pixPBlock = left.cols * windSize;

	unsigned int* disps = new unsigned int[numCalcs*m];
	fill(disps, disps + numCalcs * m, UINT_MAX);

	cudaMalloc(&d_disparity, dispImage.step * dispImage.rows);
	cudaMalloc(&d_left, left.step * left.rows);
	cudaMalloc(&d_right, right.step * right.rows);
	cudaMalloc(&d_disps, sizeof(unsigned int) * numCalcs * m);

	cudaMemcpy(d_disparity, dispImage.ptr(), dispImage.step * dispImage.rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_left, left.ptr(), left.step * left.rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_right, right.ptr(), right.step * right.rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_disps, disps, sizeof(unsigned int) * numCalcs * m, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	disparity<<<m, 1024, (sizeof(unsigned char) * 2 * (left.cols) * windSize) >>>(d_disparity, d_left, d_right, m, n, windSize, numCalcs, pixPBlock, d_disps);
	cudaEventRecord(stop);


	cudaMemcpy(dispImage.ptr(), d_disparity, dispImage.step * dispImage.rows, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "GPU Time ms: " << milliseconds << endl;


	cudaFree(d_disparity);
	cudaFree(d_left);
	cudaFree(d_right);
	cudaFree(d_disps);

	delete[] disps;

}
