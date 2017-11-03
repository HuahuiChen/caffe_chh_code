#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_ 

#include <algorithmr>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

/**
 * issue (#1009, #1010)
 */

namespace boost {
    class mutex;
}

namespace caffe {

template <typename> Dtype
class layer {
  public:
    explicit Layer(const LayerParameter& param)
        : layer_param(param), is_shared(false) {
            phase_ = param.phase();
            if (layer_param_.blob_size() > 0) {
                blobs_.resize(layer_param_.blobs_size());
                for (int i = 0; i < layer_param_.blobs_size(); ++i) {
                    blobs_[i].reset(new Blob<Dtype>());
                    blobs_[i]->FromProto(layer_param_.blobs(i));
                }
            }
    }  

    virtual ~Layer() {

    }

    void SetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        InitMutex();
        CheckBlobCounts(bottom, top);
        LayerSetUp(bottom, top);
        Reshape(bottom, top);
        SetLossWeights(top);
    }

    virtual void LayerSetUp(vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    }

    virtual inline bool ShareInParallel() const {
        return false;
    }

    inline bool IsShared() const {
        return is_shared_;
    }

    inline void SetShared(bool is_shared) {
        CHECK(ShareInParallel || !is_shared)
            << type() << "Layer does not support sharing.";
        is_shared_ = is_shared;    
    }

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) = 0;
    
    inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    inline void Backward(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom);    

    vector<shared_ptr<Blob<Dtype> > >& blobs() {
        return blobs_;
    }

    const LayerParameter& layer_param() const {
        return layer_param_;
    }

    virtual void ToProto(LayerParameter param, bool write_diff = false);

    inline Dtype loss(const int top_index) const {
        return (loss_.size() > top_index ? loss_[top_index] : Dtype(0);
    }

    inline void set_loss(const int top_index, const Dtype value) {
        if (loss_.size() <= top_index) {
            loss_.resize(top_index + 1, Dtype(0));
        }
        loss_[top_index] = value;
    }

    virtual inline const char* type() const {
        return "";
    }


    virtual inline int ExactNumBottomBlobs() const {
        return -1;
    }

    virtual inline int MinBottomBlobs() const {
        return -1;
    }

    virtual inline int MaxBottomBlobs() const {
        return -1;
    }
    virtual inline int ExactNumTopBlobs() const {
        return -1;
    }

    virtual inline int MinTopBlobs() const {
        return -1;
    }

    virtual inline int MaxTopBlobs() const {
        return -1;
    }

    virtual inline bool EqualNumBottomTopBlobs() const {
        return false;
    }

    /// If this method returns true, Net::Init will create enough "anonymous" top 
    /// blobs to fulfill the requirement specified by ExactNumTopBlobs or MinTopBlobs.
    virtual inline boot AutoTopoBlobs() const {
        return false;
    }

    virtual inline bool AllowForceBackward(const int bottom_index) const {
        return true;
    }

    inline bool param_propagate_down(const int param_id) {
        return (param_propagate_down_.size() > param_id ?
            param_propagate_down_[param_id] : false;
    }

    inline void set_param_propagate_down(const int param_id, const bool value) {
        if (param_propagate_down_.size() <= param_id) {
            param_propagate_down_.resize(param_id + 1, true);
        }
        param_propagate_down_[param_id] = value;
    }

  protected:
    LayerParameter layer_param_;
    Phase phase_;
    // The vector that stores the learnable parameters as a set 
    // of blobs, like weights, bias
    vector<shared_ptr<Blob<Dtype> > > blobs_;
    vector<bool> param_propagate_down_;
    vector<Dtype> loss_;

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) = 0;
    
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        // LOG(WARNING) << "Using CPU code as backup.";
        return Forward_cpu(bottom, top);
    }    
    
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) = 0;    
      
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
        // LOG(WARNING) << "Using CPU code as backup.";
        return Backward_gpu(top, propagate_down, bottom);
    }    

    virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        if (ExactNumBottomBlobs() >= 0) {
            CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
                << type() << " Layer takes " << ExactNumBottomBlobs()
                << " bottom blobs as input.";
        }
        if (MinBottomBlobs() >= 0) {
            CHECK_LE(MinBottomBlobs(), bottom_size())
                << type() << " Layer takes at least " << MinBottomBlobs()
                << " bottom blobs as input.";
        }
        if (MaxBottomBlobs() >= 0) {
            CHECK_GE(MaxBottomBlobs(), bottom.size())
                << type() << " Layer takes at most " << MaxBottomBlobs()
                << " bottom blobs as input.";
        }

        if (ExactNumTopBlobs() >= 0) {
            CHECK_EQ(ExactNumTopBlobs(), top.size())
                << type() << " Layer takes " << ExactNumTopBlobs()
                << " top blobs as output.";
        }
        if (MinTopBlobs() >= 0) {
            CHECK_LE(MinTopBlobs(), top_size())
                << type() << " Layer takes at least " << MinTopBlobs()
                << " top blobs as output.";
        }
        if (MaxTopBlobs() >= 0) {
            CHECK_GE(MaxTopBlobs(), top.size())
                << type() << " Layer takes at most " << MaxTopBlobs()
                << " top blobs as output.";
        }
        if (EqualNumBottomTopBlobs()) {
            CHECK_EQ(bottom.size(), top.size())
                << type() << " Layer produces one top blob as output for each "
                << "bottom blob input.";
        }

    } 

   inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
       const int num_loss_weights = layer_param_.loss_weights_size();
       if (num_loss_weights) {
           CHECK_EQ(top.size(), num_loss_weights) << "loss_weights must be "
            " unspecified or specified once per top blob.";
           for (int top_id = 0; top_id < top.size(); ++top_id) {
                const Dtype loss_weight = layer_param_.loss_weight(top_id);
                if (loss_weight == Dtype(0) {
                    continue;
                }
                set_loss(top_id, loss_weight);
                const int count = top[top_id]->count();
                Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
                caffe_set(count, loss_weight, loss_multiplier);
           }  
       }
   } 
    
  private:
    bool is_shared_;
    shared_ptr<boost::mutex> forward_mutex_;
    void InitMutex();
    void Lock();
    void Unlock();
    DISABLE_COPY_AND_ASSIGN(Layer);  
}; // class layer    

template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Lock();
    Dtype loss = 0;
    Reshape(bottom, top);
    switch (Caffe::mode()) {
        case Caffe::CPU:
            Forward_cpu(bottom, top);
            for (int top_id = 0; top_id < top.size(); ++top_id) {
                if (!loss(top_id)) {
                    continue;
                }
                const int count = top[top_id]->count();
                const Dtype* data = top[top_id]->cpu_data();
                const Dtype* loss_weights = top[top_id]->cpu_diff();
                loss += caffe_cpu_dot(count, data, loss_weights);

            }
            break;
        case Caffe::GPU:
            Forward_gpu(bottom, top);
#ifndef CPU_ONLY
            for (int top_id = 0; top_id < top.size(); ++top) {
                if (!loss(top_id)) {
                    continue;
                }
                const int count = top[top_id]->count();
                const Dtype* data = top[top_id]->gpu_data();
                const Dtype* loss_weights = top[top_id]->gpu_diff();
                Dtype blob_loss = 0;
                caffe_gpu_dot(count, data, loss_weights, &blob_loss);
                loss += blob_loss;
            }                 
#endif 
            break;
        default:
            LOG(FATAL) << "Unknown caffe mode.";
                
    }
    Unlock();
    return loss;
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    switch (Caffe::mode()) {
        case Caffe::CPU:
            Backward_cpu(top, propagate_down, bottom);
            break;
        case Caffe::GPU:
            Backward_gpu(top, propagate_down, bottom);
            break;
       default:
            LOG(FATAL) << "Unknown caffe mode.";             
    }
}

template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
    param->Clear();
    param->CopyFrom(layer_param_);
    param->clear_blobs();
    for (int i = 0; i < blobs_.size(); ++i) {
        blobs_[i]->ToProto(param->add_blobs(), write_diff);
    }
}
    
}




#endif // CAFFE_LAYER_H_
