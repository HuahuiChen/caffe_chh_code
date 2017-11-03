#ifndef CAFFE_BNLL_LAYER_HPP_
#define CAFFE_BNLL_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class BNLLLayer : public NeuronLayer<Dtype> {
  public:
    explicit BNLLLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {

        }
    virtual inline const char* type() const {
        return "BNLL";
    }      
  
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);  
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);  
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vecto<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vecto<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


}; // class BNLLLayer

} // namespace caffe


#endif // CAFFE_BNLL_LAYER_HPP_j
