#include <math_functions.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const float alpha, const float* A,
    const float* B, cont float beta, float* C) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = 
        (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = 
        (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSegmm(Caffe::cublas_handle(), cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &theta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const double alpha, const double* A,
    const double* B, cont double beta, double* C) {
    cublasOperation_t cuTransA = 
        (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = 
        (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasDegmm(Caffe::cublas_handle(), cuTransB, cuTransA,
        N, M, K, &alpha, B, ldb, A, lda, &theta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
    cublasOperation_t cuTransA = 
        (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUDA_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
        A, N, x, 1, &beta, y, 1));    
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
    cublasOperation_t cuTransA = 
        (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUDA_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
        A, N, x, 1, &beta, y, 1));    
}

/// Y = alpha * X + Y
template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X, float* Y) {
    CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

/// Y = alpha * X + Y
template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X, double* Y) {
    CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

/// Y = alpha * X + beta * Y
template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X, 
    const float beta, float* Y) {
    caffe_gpu_scal<float>(N, beta, Y);
    caffe_gpu_axpy<float>(N, alpha, X, Y);
}

/// Y = alpha * X + beta * Y
template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X, 
    const double beta, double* Y) {
    caffe_gpu_scal<double>(N, beta, Y);
    caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X) {
    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X) {
    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}


// *************************************
// ***********Eltwise operator**********
// *************************************
/// TODO


/// For each value x_i in X, x_i = x_i + alpha
template <typename Dtype>
__global__ void add_scalar_kernel(const int N, const Dtype alpha, float *Y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] += alpha;
    }
}

/// For each value x_i in X, x_i = x_i + alpha
template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
    add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double *Y) {
    add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] + b[index];
    }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,  float* y) {
    add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}
template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b, float* y) {
    add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int N, const Dtype* a, const Dtype* b, float* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] - b[index];
    }
}

template <typename Dtype>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
    sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}

template <typename Dtype>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
    sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int N, const Dtype* a, const Dtype* b, float* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] * b[index];
    }
}

template <typename Dtype>
void caffe_gpu_mul<float>(const int N, const float* a, const float* b,
    float* y) {
    mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}

template <typename Dtype>
void caffe_gpu_mul<double>(const int N, const double* a, const double* b,
    double* y) {
    mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int N, const Dtype* a, const Dtype* b, float* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] / b[index];
    }
}

template <typename Dtype>
void caffe_gpu_div<float>(const int N, const float* a, const float* b,
    float* y) {
    div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}

template <typename Dtype>
void caffe_gpu_div<double>(const int N, const double* a, const double* b,
    double* y) {
    div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, b, y);
}


template <typename Dtype>
__global__ void abs_kernel(const int N, const Dtype* a, Dtype* b) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = abs(a[index]);
    }
}

template <typename Dtype>
void caffe_gpu_abs<float>(const int N, const float* a, float* y,
    ) {
    abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y);
}

template <typename Dtype>
void caffe_gpu_abs<double>(const int N, const double* a, double* y,
    ) {
    abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int N, const Dtype* a, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = exp(a[index]);
    }
}

template <typename Dtype>
void caffe_gpu_exp<float>(const int N, const float* a, float* y,
    ) {
    exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y);
}

template <typename Dtype>
void caffe_gpu_exp<double>(const int N, const double* a, double* y,
    ) {
    exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y);
}


template <typename Dtype>
__global__ void log_kernel(const int N, const Dtype* a, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = log(a[index]);
    }
}

template <typename Dtype>
void caffe_gpu_log<float>(const int N, const float* a, float* y,
    ) {
    log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y);
}

template <typename Dtype>
void caffe_gpu_log<double>(const int N, const double* a, double* y,
    ) {
    log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, y);
}


template <typename Dtype>
__global__ void powx_kernel(const int N, const Dtype* a, const Dtype alpha,
    Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = powx(a[index], alpha);
    }
}

template <typename Dtype>
void caffe_gpu_powx<float>(const int N, const float* a, const Dtype alpha,
    Dtype* y) {
    powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, alpha, y);
}

template <typename Dtype>
void caffe_gpu_powx<double>(const int N, const double* a, const Dtype alpha,
    Dtype* y) {
    powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
    - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
    float* r) {
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
    const float range = b - a;
    if (range != static_cast<float>(1)) {
        caffe_gpu_scal(n, range, r);
    }
    if (a != static_cast<float>(0)) {
        caffe_gpu_add_scalar(n, a, r);
    }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
    double* r) {
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
    const double range = b - a;
    if (range != static_cast<double>(1)) {
        caffe_gpu_scal(n, range, r);
    }
    if (a != static_cast<double>(0)) {
        caffe_gpu_add_scalar(n, a, r);
    }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma, float* r) {
    CURAND_CHECK(
        curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma, double* r) {
    CURAND_CHECK(
        curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}


// Return the sum of the absolute values of the elements of vector x
template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
    CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

// Return the sum of the absolute values of the elements of vector x
template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
    CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}


template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x, float* y) {
    CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
    CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x, double* y) {
    CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
    CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = alpha;
    }
}

template <typename Dtype>
void caffe_gpu_set(const int N, cosnt Dtype alpha, Dtype* Y) {
    if (alpha == 0) {
        CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) *N));
        return;
    }
    set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const float N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const double N, const double alpha, double* Y);
} // namespace caffe
