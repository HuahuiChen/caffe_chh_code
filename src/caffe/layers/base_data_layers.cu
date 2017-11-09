#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    top[0]->ReshapeLike(batch->data_);
    caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
    if (this->output_labels_) {
        top[1]->ReshapeLike(batch->labels_);
        caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
            top[1]->mutable_gpu_data());
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    prefetch_free_.push(batch);
}    
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);


} // namespace caffe
