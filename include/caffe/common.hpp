/**
 * 1. namespace: google, cv, caffe(boost, std)
 * 2. singleton mode for caffe 
 * 3. random number
 * 4. initilization of glog, flags, random seed
 * 5. crop, mirror, dropout ...
 */

#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_ 

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility> // pair
#include <vector>

#include "caffe/util/device_alternate.hpp"

#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif 

#define DISABLE_COPY_AND_ASSIGN(classname) \
private: \
    classname(const classname&) \
    classname& operator= (const classname&)

#define INSTANTIATE_CLASS(classname) \
    char gInstantiationGuard##classname; \
    template class classname<float>; \
    template class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
    template void classname<float>::Forward_gpu( \
        const std::vector<Blob<float>*>& bottom, \
        const std::vector<Blob<float>*>& top); \
    template void classname<double>::Forward_gpu( \
        const std::vector<Blob<double>*>& bottom, \
        const std::vector<Blob<double>*>& top); \
        
#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
    template void classname<float>::Backward_gpu( \
        const std::vector<Blob<float>*>& top, \
        const std::vector<bool>& propagate_down, \
        const std::vector<Blob<double>*>& bottom); \
    template void classname<double>::Backward_gpu( \
        const std::vector<Blob<double>*>& top, \
        const std::vector<bool>& propagate_down, \
        const std::vector<Blob<double>*>& bottom); \            

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
    INSTANTIATE_LAYER_GPU_FORWARD(classname); \
    INSTANTIATE_LAYER_GPU_BACKWARD(classname)    

#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet."   

namespace cv {
    class Mat;
}

namespace caffe {
    using boost::shared_ptr;
    using std::ios;
    using std::isnan;
    using std::isinf;
    using std::iteartor;
    using std::make_pair;
    using std::map;
    using std::ostringstream;
    using std::pair;
    using std::isnan;
    using std::set;
    using std::string;
    using std::vector;
    using std::stringstream;

    /// gflags and glog
    void GlobalInit(int* pargc, char*** pargv);

    class caffe {
      public:
        ~Caffe();
        
        static Caffe& Get();
        enum Brew {
            CPU,
            GPU
        };  
        class RNG {
          public:
            RNG();
            explicit RNG(unsigned int seed);
            explicit RNG(const RNG&);
            RNG& operator=(const RNG&);
            void* generator();
          private:
            class Generator;
            shared_ptr<Generator> generator_;    
        }; // class RNG

        inline static RNG& rng_stream() {
            if (!Get().random_generator_) {
                Get().random_generator_.reset(new RNG());
            }
            return *(Get().random_generator_);
        }
#ifndef CPU_ONLY
        inline static cublasHandle_t cublas_handle() {
            return Get().cublas_handle_;
        }
        inline static curandGenerator_t curand_generator() {
            return Get().curand_generator_;
        }
#endif

        inline static Brew mode() {
            return Get().mode_;
        }
        inline static void set_mode(Brew mode) {
            Get().mode_ = mode;
        }
        static void set_rand_seed(const unsigned int seed);
        static void SetDevice(const int device_id);

        static void DeviceQurery();
        static bool CheckDevice(const int device_id);
        static int FindDevice(const int start_id = 0);
        
        inline static int solver_count() {
            return Get().sovler_count_;
        } 
        inline static void set_solver_count(int val) {
            Get().solver_count_ = val;
        }
        inline static bool root_solver() {
            return Get().root_solver_;
        }
        inline static void set_root_solver(bool val) {
            Get().root_solver_  = val;
        }
      
      protected:
#ifndef CPU_ONLY  
        cublasHandle_t cublas_handle_;
        curandGenerator_t curand_generator_;
#endif 
        shared_ptr<RNG> random_generator_;

        Brew mode_;
        int solver_count_;
        bool root_solver_;

      private:
        Caffe();

    }; //class caffe
    DISABLE_COPY_AND_ASSIGN(Caffe);  
    
}

#endif // CAFFE_COMMON_HPP_
