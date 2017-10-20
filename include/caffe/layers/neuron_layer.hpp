#ifdef CAFFE_NEURON_LAYER_H_
#define CAFFE_NEURON_LAYER_H_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
  
  public:
    explict NeuronLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {

        }
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const {
        return 1;
    }    
    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }

}; // class NeuronLayer


} // namespace caffe


#endif // CAFFE_NEURON_LAYER_H_
