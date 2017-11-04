#ifndef CAFFE_DECONV_LAYER_HPP_
#define CAFFE_DECONV_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class DeconvlutionLayer : public BaseConvolutionLayer<Dtype> {
  public:
    explicit DeconvlutionLayer(const LayerParameter& param) 
        : BaseConvolutionLayer<Dtype>(param) {

        }
    virtual inline const char* type() {
        return "Deconvolution";
    }    
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual inline bool reverse_dimensions() {
        return true;
    }
    virtual void compute_output_shape();
      

}; // class deconv

} // namespace caffe

#endif // CAFFE_DECONV_LAYER_HPP_
