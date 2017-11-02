#ifndef CAFFE_BASE_DATA_LAYER_HPP_
#define CAFFE_BASE_DATA_LAYER_HPP_ 

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {

  public:
    explicit BaseDataLayer(const LayerParameter& param);
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual inline bool ShareInParallel() const {
        return true;
    }
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    }
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {

    }

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }
     
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }
  protected:
    TransformationParameter transform_param_;
    shared_ptr<DataTransformer<Dtype> > data_transformer_;
    bool output_labels_;  

}; // class BaseDataLayer

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
  public:
    explicit BasePrefetchingDataLayer(const LayerParameter& param);
    void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    static const int PREFETCH_COUNT = 3;
  protected:
    virtual void InternalThreadEntry();  
    virtual void load_batch(Batch<Dtype>& batch) = 0;
    Batch<Dtype> prefetch_[PREFETCH_COUNT];
    BlockingQueue<Batch<Dtype>*> prefetch_free_;
    BlockingQueue<Batch<Dtype>*> prefetch_full_;

    Blob<Dtype> transformed_data_;
}; // class BasePrefetchingDataLayer

} // namespace caffe


#endif // CAFFE_BASE_DATA_LAYER_HPP_
