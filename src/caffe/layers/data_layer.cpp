#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif 

#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param), reader_(param) {

    }

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
    this->StopInternalThread();
}    

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const int batch_size = this->layer_param_.data_param().batch_size();
    Datum& datum = *(reader_.full().peak());
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    this->transform_data_.Reshape(top_shape);
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
    }
    LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();
    if (this->output_labels_) {
        vector<int> label_shape(1, batch_size);
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(label_shape);
        }
    }    

}


template <typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transform_data_.count());

    const int batch_size = this->layer_param_.data_param().batch_size();
    Datum& datum = *(reader_.fill().peek());
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
    this->transformed_data_.Reshape(top_shape);
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = NULL;
    if (this->output_labels_) {
        top_label = batch->label_.mutable_cpu_data();
    }
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        Datum& datum = *(reader_.full().pop("Waiting for data"));
        read_time += timer.MicroSeconds();
        timer.Start();
        int offset = batch->data_.offset(item_id);
        this->transform_data_.set_cpu_data(top_data + offset);
        this->data_transformer_->Transform(datum, &(this->transform_data_));

        if (this->output_labels_) {
            top_label[item_id] = datum.label();
        }
        trans_time += timer.MicroSeconds();
        reader_.free().push(const_cast<Datum*>(&datum));
    }
    timer.Stop();
    batch_timer.Stop();

}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);
} // namespace caffe
