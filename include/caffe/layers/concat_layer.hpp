#ifndef CAFFE_CONCAT_LAYER_HPP_
#define CAFFE_CONCAT_LAYER_HPP_ 

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ConCatLayer : public Layer<Dtype> {
  public:
    explicit ConCatLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {

        }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const {
        return "Concat";
    }    
    virtual inline int MinBottomBlobs() const {
        return 1;
    }

    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }
  
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);  
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);  
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    int count_;
    int num_concats_;
    int concat_input_size_;
    int concat_axis_;

}; // class ConCatLayer


} // namespace caffe


#endif // CAFFE_CONCAT_LAYER_HPP_

