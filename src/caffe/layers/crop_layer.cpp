#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const CropParameter& param = this->layer_param_.crop_param();
    CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
    int input_dim = bottom[0]->num_axes();
    const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
    CHECK_LT(start_axis, input_dim) << "crop axis bigger than input dim";
    if (param.offset_size() > 1) {
        CHECK_EQ(start_axis + param.offset_size(), input_dim)
            << "number of offset values specified must be equal to the number of  "
            << "dimensions following axis.";
    }
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const CropParameter& param = this->layer_param_.crop_param()j;
    int input_dim = bottom[0]->axes();
    const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
    offsets = vector<int>(input_dim, 0);
    vector<int> new_shape(bottom[0]->shape());

    for (int i = 0; i < input_dim; ++i) {
        int crop_offset = 0;
        int new_size = bottom[0]->shape(i);
        if (i >= start_axis) {
            new_size = bottom[1]->shape(i);
            if (param.offset_size() == 1) {
                crop_offset = param.offset(0);
            } else if (param.offset_size() > 1) {
                crop_offset = param.offset(i - start_axis);
            }
        }
        CHECK_GE(bottom[0]->shape(i) - crop_offset,
            bottom[1]->shape(i))
            << "invalid crop parameters in dimension: " << i;
        new_shape[i] = new_size;
        offsets[i] = crop_offset;    
    }
    top[0]->Reshape(new_shape);
}

template <typename Dtype>
void CropLayer<Dtype>::crop_copy(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top, const vector<int>& offset,
    vector<int> indices, int cur_dim, const Dtype* src_data,
    Dtype* dest_data, bool is_forward) {
    if (cur_dim + 1 < top[0]->num_axes()) {
        for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
            indices[cur_dim] = i;
            crop_copy(bottom, top, offsets, indices, cur_dim + 1,
                src_data, dest_data, is_forward);
        }
    } else {
        for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
            std::vector<int> ind_red(cur_dim, 0);
            std::vector<int> ind_off(cur_dim + 1, 0);
            for (int j = 0;j < cur_dim; ++j) {
                ind_red[j] = indices[j];
                ind_off[j] = indices[j] + offsets[j];
            }
            ind_off[cur_dim] = offsets[cur_dim];
            if (is_forward) {
                caffe_copy(top[0]->shape(cur_dim),
                    src_data + bottom[0]->offset(ind_off),
                    dest_data + top[0]->offset(ind_red));
            } else {
                caffe_copy(top[0]->shape(cur_dim),
                    src_data + top[0]->offset(ind_red),
                    dest_data + bottom[0]->offset(ind_off));
            }
        }
    }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    std::vector<int> indices(top[0]->num_axes(), 0);
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    crop_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (propagate_down[0]) {
        caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
        std::vector<int> indices(top[0]->num_axes(), 0);
        crop_copy(bottom, top, offsets, indices, 0, top_diff, bottom_diff, false);
    }
}

#ifdef CPU_ONLY 
STUB_GPU(CropLayer)
#endif 

INSTANTIATE_CLASS(CropLayer);
REGISTER_LAYER_CLASS(Crop);
} // namespace caffe
