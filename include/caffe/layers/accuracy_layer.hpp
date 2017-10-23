#ifndef CAFFE_CAFFE_ACCURACY_LAYER_HPP_
#define CAFFE_CAFFE_ACCURACY_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
  public:
    explicit AccuracyLayer(const LayerParameter param) :
        Layer<Dtype>(param) {

        } 
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual inline const char*() type() const {
        return "Accuracy";
    }    
    virtual inline int ExactNumBottomBlobs() const {
        return 2;
    }
    // If there ara two top blobs, then the second blob will contain
    // accuracies per class.
    virtual inline int MinTopBlobs() const {
        return 1;
    }
    virtual inline int MaxTopBlobs() const {
        return 2;
    }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top);   
    virtual void Backward_cpu(const vector<Blob<Dtype>*>&  top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) {
                NOT_IMPLEMENTED;
            }
        }
    }

    int label_axis_;
    int outer_num_;
    int inner_num_;

    int top_k_;

    bool has_ignore_label_;
    int ignore_label_;

    Blob<Dtype> nums_buffer_;

}; // class AccuracyLayer


} // namespace caffe


#endif // CAFFE_CAFFE_ACCURACY_LAYER_HPP_
