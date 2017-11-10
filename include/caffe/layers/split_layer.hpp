#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    
template <typename Dtype>
class SplitLayer : public Layer<Dtype> {

  public:
    explicit SplitLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {

        }  
    virtual inline const char* type() const  {
        return "Split";
    }    
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const {
        return 1;
    }
    virtual inline int MinTopBlobs() const {
        return 1;
    }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom);  
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom);  
    int count_;

};    
} // namespace caffe


#endif // CAFFE_SPLIT_LAYER_HPP_
