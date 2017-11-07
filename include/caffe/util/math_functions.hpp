#ifndef CAFFE_UTIL_MATH_FUNCTIONS_HPP_
#define CAFFE_UTIL_MATH_FUNCTIONS_HPP_ 

#include <stdint.h>
#include <cmath>

#include "glog/logging.h"
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"


// Function analysis:
// http://blog.csdn.net/seven_first/article/details/47378697 
namespace caffe {

/// C = alpha * A * B + beta * C
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

/// y = alpha * A * x + beta * y
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

/// Y = alpha * X + Y
template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

/// Y = alpha * X + beta * Y
template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X, 
    const Dtype beta, Dtype* Y);

/// Y = X
template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y);
    
/// X = alpha. NxN size
template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* X);

/// What's difference between caffe_set and caffe_memset ?
inline void caffe_memset(const size_t N, const int alpha, void* X) {
    memset(X, alpha, N); // NOLINT(caffe/alt_fn)
}


// *************************************
// ***********Eltwise operator**********
// *************************************

/// For each value x_i in X, x_i = x_i + alpha
template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

/// For each value x_i in X, x_i = x_i * alpha
template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

/// For each value y_i in y, y_i = y_i * y_i 
template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a , Dtype* y);

/// For each value y_i in y, y_i = a_i + b_i 
template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i - b_i 
template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i * b_i 
template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i / b_i 
template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i ^ b 
template <typename Dtype>
void caffe_powx(const int N, const Dtype* a, const Dtype b, Dtype *y);


// *************************************
// ***********Random function***********
// *************************************

unsigned int caffe_rng_rand();

// B最大方向上的可以表示的最接近的数,没看懂
template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma, Dtype* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);

// return x .* y
template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

// Return the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

template <typename Dtype>
inline int8_t caffe_sign(Dtype val) {
    return (Dtype(0) < val) - (val < Dtype(0));
}

#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
    template <typename Dtype> \
    void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
        CHECK_GT(n, 0); CHECK(x); CHECK(y); \
        for (int i = 0; i < n; ++i) { \
            operation; \
        } \
    } 

DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));    

DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);


#ifndef CPU_ONLY

/// C = alpha * A * B + beta * C
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const Dtype alpha, const Dtype* A,
    const Dtype* B, const Dtype beta, Dtype* C);

/// y = alpha * A * x + beta * y
template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);

/// Y = alpha * X + Y
template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

/// Y = alpha * X + beta * Y
template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X, 
    const Dtype beta, Dtype* Y);

/// Y = X
template <typename Dtype>
void caffe_gpu_memcpy(const size_t N, const void* X, void* Y);
    
/// X = alpha. NxN size
template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* X);

/// What's difference between caffe_set and caffe_memset ?
inline void caffe_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY 
    CUDA_CHECK(cudaMemset(X, alpha, N));
#else 
    NO_GPU;
#endif
}

/// For each value x_i in X, x_i = x_i + alpha
template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

/// For each value x_i in X, x_i = x_i * alpha
template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

/// For each value y_i in y, y_i = y_i * y_i 
template <typename Dtype>
void caffe_gpu_sqr(const int N, const Dtype* a , Dtype* y);

/// For each value y_i in y, y_i = a_i + b_i 
template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i - b_i 
template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i * b_i 
template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i / b_i 
template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype *y);

/// For each value y_i in y, y_i = a_i ^ b 
template <typename Dtype>
void caffe_gpu_powx(const int N, const Dtype* a, const Dtype b, Dtype *y);


// *************************************
// ***********Random function***********
// *************************************

unsigned int caffe_gpu_rng_rand();


template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

// y = sum(|x_i|)
template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);


template <typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype* x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template <typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
    CUDA_KERNEL_LOOP(index, n) { \
        operation; \
    } \
}\
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
    name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const float* x, float* y) { \
    name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, x, y); \
} 
#endif


} // namespace caffe



#endif // CAFFE_UTIL_MATH_FUNCTIONS_HPP_

