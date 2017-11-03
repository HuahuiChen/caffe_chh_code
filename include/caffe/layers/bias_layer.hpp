#ifndef CAFFE_BIAS_LAYER_HPP_
#define CAFFE_BIAS_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BiasLayer : public Layer<Dtype> {
  public:
    explicit BiasLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {

        }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const {
        return "Bias";
    }    
    virtual inline int MinBottomBlobs() const {
        returan 1;
    }
    virtual inline int MaxBottomBlobs() const {
        returan 2;
    }
    virtual inline int ExactNumTopBlobs() const {
        returan 1;
    }
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool> propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool> propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  private:
    Blob<Dtype> bias_multiplier_;
    int outer_dim_, bias_dim_, inner_dim_, dim_;  
    

}; //class BiasLayer


} // namespace caffe


#endif // CAFFE_BIAS_LAYER_HPP_
