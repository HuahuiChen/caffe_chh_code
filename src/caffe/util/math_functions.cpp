#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


template <>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const float alpha, const float* A,
    const float* B, cont float beta, float* C) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
        ldb, beta, C, N);
}
template <>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const double alpha, const double* A,
    const double* B, cont double beta, double* C) {
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
        ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
    cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
    cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

/// Y = alpha * X + Y
template <>
void caffe_axpy(const int N, const float alpha, const float* X, float* Y) {
    cblas_saxpy(N, alpha, X, 1, Y, 1);
}

/// Y = alpha * X + beta * Y
template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X, 
    const float beta, float* Y) {
    cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
    if (X != Y) {
        if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY 
            CUDA_CHECK(cuda_Memcpy(Y, X, size(Dtype) * N, cudaMemcpyDefault));
#else 
            NO_GPU;
#endif 
        } else {
            memcpy(Y, X, sizeof(Dtype) * N);
        }
    }
}
/// Y = X

template void caffe_copy(const int N, const int* X, int* Y);
template void caffe_copy(const unsigned int N, const unsigned int* X, unsigned int* Y);
template void caffe_copy(const int N, const float* X, float* Y);
template void caffe_copy(const int N, const double* X, double* Y);
    
/// X = alpha. NxN size
template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* X) {
    if (alpha == 0) {
        memset(Y, 0, sizeof(Dtype) * N);
        return;
    }
    for (int i = 0; i < N; ++i) {
        Y[i] = alpha;
    }
}

template void caffe_set(const int N, const int alpha, int* X);
template void caffe_set(const float N, const float alpha, float* X);
template void caffe_set(const double N, const double alpha, double* X);
/// What's difference between caffe_set and caffe_memset ?
inline void caffe_memset(const size_t N, const int alpha, void* X) {
    memset(X, alpha, N); // NOLINT(caffe/alt_fn)
}


// *************************************
// ***********Eltwise operator**********
// *************************************

/// For each value x_i in X, x_i = x_i + alpha
template <>
void caffe_add_scalar(const int N, const float alpha, float *X) {
    for (int i = 0; i < N; ++i) {
        Y[i] += alpha;
    }
}

/// For each value x_i in X, x_i = x_i * alpha
template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
    cblas_sscal(N, alpha, X, 1);
}

/// For each value y_i in y, y_i = y_i * y_i 
template <>
void caffe_sqr<float>(const int N, const float* a , float* y) {
    vsSqr(n, a, y);
}

/// For each value y_i in y, y_i = a_i + b_i 
template <>
void caffe_add<float>(const int N, const float* a, const float* b, float *y) {
    vsAdd(n, a, b, y);
}

/// For each value y_i in y, y_i = a_i - b_i 
template <>
void caffe_sub<float>(const int N, const float* a, const float* b, float *y) {
    vsSub(n, a, b, y);
}

/// For each value y_i in y, y_i = a_i * b_i 
template <>
void caffe_mul<float>(const int N, const float* a, const float* b, float *y) {
    vsMul(n, a, b, y);
}

/// For each value y_i in y, y_i = a_i / b_i 
template <>
void caffe_div<float>(const int N, const float* a, const float* b, float *y) {
    vsDiv(n, a, b, y);
}

/// For each value y_i in y, y_i = a_i ^ b 
template <>
void caffe_powx<float>(const int N, const float* a, const float b, float *y) {
    vsPowx(n, a, b, y);
}


template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
    vsExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
    vsLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}
// *************************************
// ***********Random function***********
// *************************************

unsigned int caffe_rng_rand() {
    return (*caffe_rng())();
}

// B最大方向上的可以表示的最接近的数,没看懂
template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
    return boost::math::caffe_nextafter<Dtype>(
        b, std::numeric_limits<Dtype>::max());
}

template float caffe_nextafter(const float b); 
template double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r){
    CHECK_GE(n, 0);
    CHECK(r);
//TODO
}

template <>
void caffe_rng_gaussian(const int n, const float mu, const float sigma, float* r);

template <>
void caffe_rng_bernoulli(const int n, const float p, int* r);

template <>
void caffe_rng_bernoulli(const int n, const float p, unsigned int* r);


// return x .* y
template <>
float caffe_cpu_dot(const int n, const float* x, float* y);

template <>
float caffe_cpu_strided_dot(const int n, const float* x, const int incx,
    const float* y, const int incy);

// Return the sum of the absolute values of the elements of vector x
template <>
float caffe_cpu_asum(const int n, const float* x);

template <>
inline int8_t caffe_sign(float val) {
    return (float(0) < val) - (val < float(0));
}

#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
    template <> \
    void caffe_cpu_##name(const int n, const float* x, float* y) { \
        CHECK_GT(n, 0); CHECK(x); CHECK(y); \
        for (int i = 0; i < n; ++i) { \
            operation; \
        } \
    } 

DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<float>(x[i]));    

DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template <>
void caffe_cpu_scale(const int n, const float alpha, const float *x, float* y);


#ifndef CPU_ONLY

/// C = alpha * A * B + beta * C
template <>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const float alpha, const float* A,
    const float* B, const float beta, float* C);

/// y = alpha * A * x + beta * y
template <>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta, float* y);

/// Y = alpha * X + Y
template <>
void caffe_gpu_axpy(const int N, const float alpha, const float* X, float* Y);

/// Y = alpha * X + beta * Y
template <>
void caffe_gpu_axpby(const int N, const float alpha, const float* X, 
    const float beta, float* Y);

/// Y = X
template <>
void caffe_gpu_memcpy(const size_t N, const void* X, void* Y);
    
/// X = alpha. NxN size
template <>
void caffe_gpu_set(const int N, const float alpha, float* X);

/// What's difference between caffe_set and caffe_memset ?
inline void caffe_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY 
    CUDA_CHECK(cudaMemset(X, alpha, N));
#else 
    NO_GPU;
#endif
}

/// For each value x_i in X, x_i = x_i + alpha
template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float *X);

/// For each value x_i in X, x_i = x_i * alpha
template <>
void caffe_gpu_scal(const int N, const float alpha, float *X);

/// For each value y_i in y, y_i = y_i * y_i 
template <>
void caffe_gpu_sqr(const int N, const float* a , float* y);

/// For each value y_i in y, y_i = a_i + b_i 
template <>
void caffe_gpu_add(const int N, const float* a, const float* b, float *y);

/// For each value y_i in y, y_i = a_i - b_i 
template <>
void caffe_gpu_sub(const int N, const float* a, const float* b, float *y);

/// For each value y_i in y, y_i = a_i * b_i 
template <>
void caffe_gpu_mul(const int N, const float* a, const float* b, float *y);

/// For each value y_i in y, y_i = a_i / b_i 
template <>
void caffe_gpu_div(const int N, const float* a, const float* b, float *y);

/// For each value y_i in y, y_i = a_i ^ b 
template <>
void caffe_gpu_powx(const int N, const float* a, const float b, float *y);


// *************************************
// ***********Random function***********
// *************************************

unsigned int caffe_gpu_rng_rand();


template <>
void caffe_gpu_rng_uniform(const int n, const float a, const float b, float* r);

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma, float* r);

template <>
void caffe_gpu_rng_bernoulli(const int n, const float p, int* r);

template <>
void caffe_gpu_rng_bernoulli(const int n, const float p, unsigned int* r);

template <>
void caffe_gpu_exp(const int n, const float* a, float* y);

template <>
void caffe_gpu_log(const int n, const float* a, float* y);

template <>
void caffe_gpu_abs(const int n, const float* a, float* y);

// y = sum(|x_i|)
template <>
void caffe_gpu_asum(const int n, const float* x, float* y);

template <>
void caffe_gpu_sign(const int n, const float* x, float* y);


template <>
void caffe_gpu_sgnbit(const int n, const float* x, float* y);

template <>
void caffe_gpu_fabs(const int n, const float* x, float* y);

template <>
void caffe_gpu_scale(const int n, const float alpha, const float* x, float* y);

} // namespace caffe
