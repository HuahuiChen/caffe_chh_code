#ifndef CAFFE_ABSVAL_LAYER_HPP_
#define CAFFE_ABSVAL_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class AbsValLayer : public NeuronLayer<Dtype> {

  public:
    explicit AbsValLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {

        }
    virtual void inline const char* type() const {
        return "AbsVal";
    }    

    virtual void LayerSetup(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const {
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
        const vector<Blob<bool> >& propagate_down,
        const vector<Blob<Dtype>*>& bottom);  
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<Blob<bool> >& propagate_down,
        const vector<Blob<Dtype>*>& bottom);  

};

} // namespace caffe

#endif // CAFFE_ABSVAL_LAYER_HPP_
