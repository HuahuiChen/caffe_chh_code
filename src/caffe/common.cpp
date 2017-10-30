#include <boost/thread.hpp>
#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

static boost::thread_specific_ptr<Caffe> thread_instance_;

Caffe& Caffe::Get() {
    if (!thread_instance_.get()) {
        thread_instance_.reset(new Caffe());
    }
    return *(thread_instance_.get());
}

int64_t cluster_seedgen(void) {
    int64_t s, seed, pid;
    FILE* f  = fopen("dev/urandom", "rb");
    if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
        fclose(f);
        return seed;
    }

    LOG(INFO) << "System entropy source not available, "
        " using fallback algorithm to generate seed insteed.";
    if (f)
        fclose(f);
    pid  = getpid();
    s = tim(NULL);
    seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
    return seed;            
}

void GlobalInit(int* pargc, char*** pargv) {
    ::google::ParseCommandLineFlags(pargc, pargv, true);
    ::google::InitGoogleLogging(*(pargv)[0]);
    ::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY 
Caffe::Caffe()
    : random_generator_(), mode(Caffe::CPU),
        solver_count_(1), root_solver_(true) {

        }
Caffe::~Caffe() {

}        

void Caffe::set_rand_seed(const unsigned int seed) {
    Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
    NO_GPU;
}

void Caffe::DeviceQurery() {
    NO_GPU;
}

bool Caffe::CheckDevice(const int device_id) {
    NO_GPU;
    return false;
}

int Caffe::FindDevice(const int start_id) {
    NO_GPU;
    return -1;
}

class Caffe::RNG::Generator {
  public:
    Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {
    explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {

    }
    caffe::rng_t* rng() {
        return rng_.get();
    }
  private:
    shared_ptr<caffe::rng_t> rng_;  
    }    

};

Caffe::RNG::RNG() : generator_(new Generator()) {

}

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)){

}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
    generator_ = other.generator_;
    return *this;
}


void* Caffe::RNG::generator() {
    return static_cast<void*>(generator_->rng());
}
#else
Caffe::Caffe()
    : cublas_handle_(NULL), curand_generator_(NULL), random_generator_(),
      mode_(Caffe::CPU), solver_count_(1), root_solver_(true) {
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Cannot create cublas handle. cublas won't be available.";
    }
    if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
           != CURAND_STATUS_SUCCESS || 
           curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
           != CURAND_STATUS_SUCCESS) {
        LOG(ERROR) << "Cannot create curand generator. Curand won't be available.";
    }
        
}

Caffe::~Caffe() {
    if (cublas_handle_) CUBLAS_STATUS_CHECK(cublasDestroy(cublas_handle_));
    if (curand_generator_) CURAND_CHECK(curandDesstroyGenerator(curand_generator_));
}

void Caffe::set_rand_seed(const unsigned int seed) {
    static bool g_curand_availability_logged = false;
    if (Get().curand_generator_) {
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(), seed));
        CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
    } else {
        if (!g_curand_availability_logged) {
            LOG(ERROR) << 
                " Curand not available. Skipping setting the curand seed.";
             g_curand_availability_logged = true;    
        }
    }
    GET().random_generator_.reset(new RNG(seed));
}


void Caffe::SetDevice(const int device_id) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device);
    if (current_devic == device_id) {
        return;
    } 
    CUDA_CHECK(cudaSetDevice(device_id));
    if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
    if (Get().curand_generator_) {
        CURAND_CHECK(curandDesstroyGenerator(Get().curand_generator_));
    }
    CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
    CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_, 
        CURAND_RNG_PSEUDO_DEFAULT);
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
            cluster_seedgen()));
}

void Caffe::DeviceQuery() {
    cudaDeviceProp prop;
    int device;
    if (cudaSuccess != cudaGetDevice(&device)) {
        std::cout << "No cuda device present." << std::endl;
        return;
    }
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    LOG(INFO) << "Device id:                                   " << device;
    LOG(INFO) << "Major revision number                        " << prop.major;
    LOG(INFO) << "Minor revision number                        " << prop.minor;
    LOG(INFO) << "Name                                         " << prop.name;
    LOG(INFO) << "Total global memory:                         " << prop.totalGlobalMem;
    LOG(INFO) << "Total shared memory per block:               " << prop.sharedMemPerBlock;
    LOG(INFO) << "Total registers per block                    " << prop.regsPerBlock;
    LOG(INFO) << "Warp size:                                   " << prop.warpSize;
    LOG(INFO) << "Maximum memory pitch                         " << prop.memPitch;
    LOG(INFO) << "Maximum threads per block:                   " << prop.maxThreadsPerBlock;
    LOG(INFO) << "Maximum dimension of block:                  " << prop.maxThreadsDim[0]
        << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2];
    LOG(INFO) << "Maximum dimension of grid:                   " << prop.maxGridSize[0]
        << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2];
    LOG(INFO) << "Clock rate:                                  " << prop.clockRate;
    LOG(INFO) << "Total constant memory                        " << prop.totalConstMen;
    LOG(INFO) << "Texture alignment                            " << prop.textureAlignment;
    LOG(INFO) << "Concurrent copy and execution:               " << (prop.deviceOverlap ? 
        "Yes " : "No");
    LOG(INFO) << "Number of multiprocesseors:                  " << prop.memProcessorCount;
    LOG(INFO) << "Kernel execution timeout:                    " << 
        << (prop.kernelExecTimeoutEnabled ? "Yes" : "No"); 
    return;    
}

bool Caffe::ChechDevice(const int device_id) {
    bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
              (cudaSuccess == cudaFree(0)));
    cudaGetLastError();
    return r;
}


int Caffe::FindDevice(const int device_id) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
       if (CheckDevice(i))
           return i; 
    }
    return -1;
}

class Caffe::RNG::Generator {
  public:
    Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {

    }
    explicit Generator(unsigned int seed) : rng_(new caffe::cluster_seedgen(seed)) {

    }
    caffe::rng_t* rng() {
        return rng_.get();
    }
  private:
    shared_ptr<caffe::rng_t> rng_;  
      
}; // class Caffe::RNG::Generator

Caffe::RNG::RNG() : generator_(new Generator()) {

}

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {

}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
    generator_.reset(other.generator_.get());
    return *this;
}

void* Caffe::RNG::generator() {
    return static_cast<void*>(generator_->rng());
}

const char* cublasGetErrorString(cublasStatus_t error) {
    switch(error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif 
#if CUDA_VERSION >= 6050
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
    switch(error) {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOC_FAILED:
            return "CURAND_STATUS_ALLOC_FAILED";
        case CURAND_STATUS_INVALID_VALUE:
            return "CURAND_STATUS_INVALID_VALUE";
        case CURAND_STATUS_TYPE_MISMATCH:
            return "CURAND_STATUS_TYPE_MISMATCH";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILUR";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "Unknown curand status";
}

#endif // CPU_ONLY

} // namespace caffe

