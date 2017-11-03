#include <vector>

#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchReindexLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(1, bottom[1]->num_axes());
    vector<int> newshape;
    newshape.push_back(bottom[1]->shape(0));
    for (int   
}


} // namespace caffe
