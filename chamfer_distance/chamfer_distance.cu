#pragma message("Using chamfer distance for 4D point clouds. Be careful!!!")

#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ 
void ChamferDistanceKernel(
	int b,
	int n,
	const float* xyzw1,
	int m,
	const float* xyzw2,
	float* result,
	int* result_i)
{
	const int batch=512;
	__shared__ float buf[batch*4];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){	// Batch No.i (xyzw1[i, :, :] & xyzw2[i, :, :])

		/* 
		* split xyzw2 into batches of size 512,  and then 
		* find the nearest neighbor in xyzw2[k2:(k2+512)] for each point in xyzw1
		*/
		for (int k2=0;k2<m;k2+=batch){			// for each xyzw2 batch

			/* 
			* cache xyzw2[k2:(k2+512)] into shared memory for each block
			*/
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*4;j+=blockDim.x){
				buf[j]=xyzw2[(i*m+k2)*4+j];
			}
			__syncthreads();

			/* 
			* for each point in xyzw1 find the nearest neighbor in xyzw2[k2:(k2+512)] 
			*/
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){ // for each point in xyzw1[i, :, :]
				float x1=xyzw1[(i*n+j)*4+0];
				float y1=xyzw1[(i*n+j)*4+1];
				float z1=xyzw1[(i*n+j)*4+2];
				float w1=xyzw1[(i*n+j)*4+3];
				int best_i=0;
				float best=0;
				int end_ka=end_k-(end_k&3);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							float x2=buf[k*4+0]-x1;
							float y2=buf[k*4+1]-y1;
							float z2=buf[k*4+2]-z1;
							float w2=buf[k*4+3]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*4+4]-x1;
							float y2=buf[k*4+5]-y1;
							float z2=buf[k*4+6]-z1;
							float w2=buf[k*4+7]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*4+8]-x1;
							float y2=buf[k*4+9]-y1;
							float z2=buf[k*4+10]-z1;
							float w2=buf[k*4+11]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*4+12]-x1;
							float y2=buf[k*4+13]-y1;
							float z2=buf[k*4+14]-z1;
							float w2=buf[k*4+15]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							float x2=buf[k*4+0]-x1;
							float y2=buf[k*4+1]-y1;
							float z2=buf[k*4+2]-z1;
							float w2=buf[k*4+3]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*4+4]-x1;
							float y2=buf[k*4+5]-y1;
							float z2=buf[k*4+6]-z1;
							float w2=buf[k*4+7]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*4+8]-x1;
							float y2=buf[k*4+9]-y1;
							float z2=buf[k*4+10]-z1;
							float w2=buf[k*4+11]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*4+12]-x1;
							float y2=buf[k*4+13]-y1;
							float z2=buf[k*4+14]-z1;
							float w2=buf[k*4+15]-w1;
							float d=x2*x2+y2*y2+z2*z2+w2*w2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*4+0]-x1;
					float y2=buf[k*4+1]-y1;
					float z2=buf[k*4+2]-z1;
					float w2=buf[k*4+3]-w1;
					float d=x2*x2+y2*y2+z2*z2+w2*w2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

void ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyzw1,
    const int m,
    const float* xyzw2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i)
{
	ChamferDistanceKernel<<<dim3(32,16,1),512>>>(b, n, xyzw1, m, xyzw2, result, result_i);
	ChamferDistanceKernel<<<dim3(32,16,1),512>>>(b, m, xyzw2, n, xyzw1, result2, result2_i);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in chamfer distance updateOutput: %s\n", cudaGetErrorString(err));
}


__global__ 
void ChamferDistanceGradKernel(
	int b, int n,
	const float* xyzw1,
	int m,
	const float* xyzw2,
	const float* grad_dist1,
	const int* idx1,
	float* grad_xyzw1,
	float* grad_xyzw2)
{
	for (int i = blockIdx.x; i<b; i += gridDim.x) {
		for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x*gridDim.y) {
			float x1=xyzw1[(i*n+j)*4+0];
			float y1=xyzw1[(i*n+j)*4+1];
			float z1=xyzw1[(i*n+j)*4+2];
			float w1=xyzw1[(i*n+j)*4+3];
			int j2=idx1[i*n+j];
			float x2=xyzw2[(i*m+j2)*4+0];
			float y2=xyzw2[(i*m+j2)*4+1];
			float z2=xyzw2[(i*m+j2)*4+2];
			float w2=xyzw2[(i*m+j2)*4+3];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyzw1[(i*n+j)*4+0]),g*(x1-x2));
			atomicAdd(&(grad_xyzw1[(i*n+j)*4+1]),g*(y1-y2));
			atomicAdd(&(grad_xyzw1[(i*n+j)*4+2]),g*(z1-z2));
			atomicAdd(&(grad_xyzw1[(i*n+j)*4+3]),g*(w1-w2));
			atomicAdd(&(grad_xyzw2[(i*m+j2)*4+0]),-(g*(x1-x2)));
			atomicAdd(&(grad_xyzw2[(i*m+j2)*4+1]),-(g*(y1-y2)));
			atomicAdd(&(grad_xyzw2[(i*m+j2)*4+2]),-(g*(z1-z2)));
			atomicAdd(&(grad_xyzw2[(i*m+j2)*4+3]),-(g*(w1-w2)));
		}
	}
}

void ChamferDistanceGradKernelLauncher(
    const int b, const int n,
    const float* xyzw1,
    const int m,
    const float* xyzw2,
    const float* grad_dist1,
    const int* idx1,
    const float* grad_dist2,
    const int* idx2,
    float* grad_xyzw1,
    float* grad_xyzw2)
{
	cudaMemset(grad_xyzw1, 0, b*n*4*sizeof(float));
	cudaMemset(grad_xyzw2, 0, b*m*4*sizeof(float));
	ChamferDistanceGradKernel<<<dim3(1,16,1), 256>>>(b, n, xyzw1, m, xyzw2, grad_dist1, idx1, grad_xyzw1, grad_xyzw2);
	ChamferDistanceGradKernel<<<dim3(1,16,1), 256>>>(b, m, xyzw2, n, xyzw1, grad_dist2, idx2, grad_xyzw2, grad_xyzw1);

	cudaError_t err = cudaGetLastError();
  	if (err != cudaSuccess)
	    printf("error in chamfer distance get grad: %s\n", cudaGetErrorString(err));
}
