#ifndef CAFFE_BATCHREINDEX_LAYER_HPP_
#define CAFFE_BATCHREINDEX_LAYER_HPP_ 

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BatchReindexLayer : public Layer<Dtype> {
  public:
    explicit BatchReindexLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {

        }
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<BLob<Dtype>*>& top);
    virtual inline const char* type() const {
        return "BatchReindex";
    }    
    virtual inline int ExactNumBottomBlobs() const {
        return 2;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);  

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  private:
    struct pair_sort_first {
        bool operator() (const std::pair<int, int>& left,
            const std::pair<int, int>& right) {
            return left.first < right.first;
        }
    };  
    void check_batch_reindex(int initial_num, int final_num,
        const Dtype* rinx_data);


};


} // namespace caffe

#endif  //CAFFE_BATCHRE
