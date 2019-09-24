#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

inline int opt_n_threads(int work_size) {
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)


inline dim3 opt_block_config(int x, int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads =
        max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);
  
    return block_config;
  }

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void group_points_kernel(int b, int c, int n, int npoints,
    int nsample,
    const float *__restrict__ points,
    const int *__restrict__ idx,
    float *__restrict__ out) {
        int batch_index = blockIdx.x;
        points += batch_index * n * c;
        idx += batch_index * npoints * nsample;
        out += batch_index * npoints * nsample * c;
        const int index = threadIdx.y * blockDim.x + threadIdx.x;
        const int stride = blockDim.y * blockDim.x;
        for (int i = index; i < c * npoints; i += stride) {
                const int l = i / npoints;
                const int j = i % npoints;
                for (int k = 0; k < nsample; ++k) {
                    int ii = idx[j * nsample + k];
                    out[(l * npoints + j) * nsample + k] = points[l * n + ii];
                }
        }
}

void group_points_cuda_forward(int b, int c, int n, int npoints, int nsample,
    const float *points, const int *idx,float *out) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
                b, c, n, npoints, nsample, points, idx, out);
            CUDA_CHECK_ERRORS();
}


// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints,
    int nsample, const float *__restrict__ grad_out, const int *__restrict__ idx,
    float *__restrict__ grad_points) {
        int batch_index = blockIdx.x;
        grad_out += batch_index * npoints * nsample * c;
        idx += batch_index * npoints * nsample;
        grad_points += batch_index * n * c;

        const int index = threadIdx.y * blockDim.x + threadIdx.x;
        const int stride = blockDim.y * blockDim.x;
        for (int i = index; i < c * npoints; i += stride) {
                const int l = i / npoints;
                const int j = i % npoints;
                    for (int k = 0; k < nsample; ++k) {
                            int ii = idx[j * nsample + k];
                            atomicAdd(grad_points + l * n + ii,
                            grad_out[(l * npoints + j) * nsample + k]);
                    }
        }
}

void group_points_cuda_backward(int b, int c, int n, int npoints,
 int nsample, const float *grad_out,const int *idx, float *grad_points) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
        b, c, n, npoints, nsample, grad_out, idx, grad_points);

    CUDA_CHECK_ERRORS();
}
