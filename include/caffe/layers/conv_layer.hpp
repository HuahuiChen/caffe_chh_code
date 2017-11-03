#ifndef CAFFE_CONV_LAYER_HPP_
#define CAFFE_CONV_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {

  public:
    explicit ConvolutionLayer(LayerParameter& param)
        : BaseConvolutionLayer<Dtype>(param) {

        }
    virtual inline const char* type() const {
        return "Convolution";
    }    
  
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
    virtual inline bool reverse_dimensions() {
        return false;
    }
    virtual void compute_out_shape();
      

};

} // namespace caffe


#endif // CAFFE_CONV_LAYER_HPP_
