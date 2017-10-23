#include <climits> // min,max
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmen.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
    vector<int> shape(4);
    shape[0] = num;
    shape[1] = channels;
    shape[2] = height;
    shape[3] = width;
    Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
    CHECK_LE(shape.size(), kMaxBlobAxes);
    count_ = 1;
    shape_.resize(shape.size());
    if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
        shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
    }
    int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
    for (int i = 0; i < shape.size(); ++i) {
        CHECK_GE(shape[i], 0);
        CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
        count_ *= shape[i];
        shape_[i] = shape[i];
        shape_data[i] = shape[i];
    }
    if (count_ > capacity_) {
        capacity_ = count_;
        data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
        diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
    CHECK_LE(shape.dim_size(), kMaxBlobAxes);
    vector<int> shape_vec(shape.dim_size());
    for (int i = 0; i < shape.dim_size(); ++i) {
        shape_vec[i] = shape.dim(i);
    }
    Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
    Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const  int num, const int channels, const int height,
    const int width) : capacity_(0) {
    Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
    : capacity_(0) {
        Reshape(shape);
    }

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
    CHECK(shape_data_);
    return (const int*) shape_data_->gpu_data();
}    
template <typename Dtype>
const int* Blob<Dtype>::cpu_shape() const {
    CHECK(shape_data_);
    return (const int*) shape_data_->cpu_data();
}    

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
    CHECK(data);
    data_->set_cpu_data(data);
}
template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
    CHECK(data_);
    return (const Dtype*) data_->gpu_data();
}
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
    CHECK(data_);
    return (const Dtype*) data_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
    CHECK(data_);
    return (const Dtype*) data_->gpu_diff();
}
template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
    CHECK(data_);
    return (const Dtype*) data_->cpu_diff();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_cpu_diff());
}
template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_gpu_diff());
}

template <typename Dtype>
Dtype* Blob<Dtype>::ShareData(const Blob& other) {
    CHECK_EQ(count_, other.count());
    data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
    CHECK_EQ(count_, other.count());
    diff_ = other.diff();
}

template <typename Dtype>
void Blob<Dtype>::Update() {
    switch (data_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            caffe_axpy<Dtype>(count_, Dtype(-1),
                static_cast<const Dtype*>(diff_->cpu_data()),
                static_cast<Dtype*>(data_->mutable_cpu_data()));  
            break;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
        caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
            static_cast<const Dtype*>(diff_->gpu_data()),
            static_cast<Dtype*>(data_->mutable_gpu_data()));
#else 
        NO_GPU;
#endif
            break;
        default:
            LOG(FATAL) << "SyncedMemory not initialized.";
    }
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
    if (!data_) {
        return 0;
    }
    switch (data_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            return caffe_cpu_asum(count_, cpu_data());
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY 
        {
            Dtype asum;
            caffe_gpu_asum(count_, gpu_data(), &asum);
            return asum;
        }            
#else 
        NO_GPU;
#endif 
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();    
    }
    return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
    if (!diff) 
        return 0;
    switch (diff_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            return caffe_cpu_asum(count_, cpu_diff());
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY 
        {
            Dtype asum;
            caffe_gpu_asum(count_, gpu_diff(), &asum);
            return asum;
        }
#else 
        NO_GPU;
#endif 
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();    
    }    
    return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
    if (!data_) {
        return 0;
    }
    Dtype sumsq;
    const Dtype* data;
    switch (data_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            data = cpu_data();
            sumsq = caffe_cpu_dot(count_, data, data); 
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY 
        {
            data = gpu_data();
            caffe_gpu_dot(count_, data, data, &sumsq);
        }            
#else 
        NO_GPU;
#endif 
        break;
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();    
    }
    return sumsq;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
    if (!diff_) {
        return 0;
    }
    Dtype sumsq;
    const Dtype* diff;
    switch (diff_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            diff = cpu_diff();
            sumsq = caffe_cpu_dot(count_, diff, diff); 
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY 
        {
            diff = gpu_diff();
            caffe_gpu_dot(count_, diff, diff, &sumsq);
        }            
#else 
        NO_GPU;
#endif 
        break;
        case SyncedMemory::UNINITIALIZED:
            return 0;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();    
    }
    return sumsq;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
    Dtype* data;
    if (!data_) {
        return;
    }
    switch (data_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            data = mutable_cpu_data();
            caffe_scal(count_, scale_factor, data);
            return;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY 
        data = mutable_gpu_data();
        caffe_gpu_scal(count_, scale_factor, data);
        return;
#else 
        NO_GPU;
#endif 
        case SyncedMemory::UNINITIALIZED:
            return;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();    
    }
}



template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
    Dtype* diff;
    if (!diff_) {
        return;
    }
    switch (diff_->head()) {
        case SyncedMemory::HEAD_AT_CPU:
            diff = mutable_cpu_diff();
            caffe_scal(count_, scale_factor, diff);
            return;
        case SyncedMemory::HEAD_AT_GPU:
        case SyncedMemory::SYNCED:
#ifndef CPU_ONLY 
        diff = mutable_gpu_diff();
        caffe_gpu_scal(count_, scale_factor, diff);
        return;
#else 
        NO_GPU;
#endif 
        case SyncedMemory::UNINITIALIZED:
            return;
        default:
            LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();    
    }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
    if (other.has_num() || other.has_channels() || 
            other.has_height() || other.has_width()){
        return shape_.size() <= 4 &&
            LegacyShape(-4) == other.num() &&
            LegacyShape(-3) == other.channels() &&
            LegacyShape(-2) == other.height() &&
            LegacyShape(-1) == other.width();
    }
    vector<int> other_shape(other.shape().dim_size());
    for (int i = 0; i < other.shape().dim_size(); ++i) {
        other_shape[i] = other.shape().dim(i);
    }
    return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
    if (source.count() != count_ || source.shape() != shape_) {
        if (reshape) {
            ReshapeLike(source);
        } else {
            LOG(FATAL) << "Trying to copy blobs of different sizes.";
        }
    }
    switch (Caffe::mode()) {
        case Caffe::GPU:
            if (copy_diff) {
                caffe_copy(count_, source.gpu_diff(),
                    static_cast<Dtype*>(diff_->mutable_gpu_data()));
            } else {
                caffe_copy(count_, source.gpu_data(),
                    static_cast<Dtype*>(data_->mutable_gpu_data()));
            }
            break;
        case Caffe::CPU:
            if (copy_diff) {
                caffe_copy(count_, source.cpu_diff(),
                    static_cast<Dtype*>(diff_->mutable_cpu_data()));
            } else {
                caffe_copy(count_, source.cpu_data(),
                    static_cast<Dtype*>(data_->mutable_cpu_data()));
            }
            break;
        default:
            LOG(FATAL) << "Unknown caffe mode.";    
    }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
    if (reshape) {
        vector<int> shape;
        if (proto.has_num() || proto.has_channels() || 
            proto.has_height() || proto.has_width()) {
            shape.resize(4);
            shape[0] = proto.num();
            shape[1] = proto.channels();
            shape[2] = proto.height();
            shape[3] = proto.width();
        } else {
            shape.resize(proto.shape().dim_size());
            for (int i = 0; i < proto.shape().dim_size(); ++i) {
                shape[i] = proto.shape().dim(i);
            }
        }
        Reshape(shape);
    } else {
        CHECK(ShapeEquals(proto)) << "shape mismatch ( reshape  not set) ";
    }
    Dtype* data_vec = mutable_cpu_data();
    if (proto.double_data_size() > 0) {
        CHECK_EQ(count_, proto.double_data_size());
        for (int i = 0; i < count_; ++i) {
            data_vec[i] = proto.double_data(i);
        }
    } else {
        CHECK_EQ(count_, proto.data_size());
        for (int i = 0; i < count_; ++i) {
            data_vec[i] = proto.data(i);
        }
    }
    if (proto.double_diff_size() > 0) {
        CHECK_EQ(count_, proto.double_diff_size());
        Dtype* diff_vec = mutable_cpu_diff();
        for (int i = 0; i < count_; ++i) {
            diff_vec[i] = proto.double_diff(i);
        }
    } else if (proto.diff_size() > 0) {
        CHECK_EQ(count_, proto.diff_size());
        Dtype* diff_vec = mutable_cpu_diff();
        for (int i = 0; i < count_; ++i) {
            diff_vec[i] = proto.diff(i);
        }
    }

}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
    proto->clear_shape();
    for (int i = 0; i < shape_.size(); ++i) {
        proto->mutable_shape()->add_dim(shape_[i]);
    }
    proto->clear_double_data();
    proto->clear_double_diff();
    const double* data_vec = cpu_data();
    for (int i = 0; i < count_; ++i) {
        proto->add_double_data(data_vec[i]);
    }
    if (write_diff) {
        const double* diff_vec = cpu_diff();
        for (int  i = 0; i < count_; ++i) {
            proto->add_double_data(diff_vec[i]);
        }
    }
}
template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
    proto->clear_shape();
    for (int i = 0; i < shape_.size(); ++i) {
        proto->mutable_shape()->add_dim(shape_[i]);
    }
    proto->clear_double_data();
    proto->clear_double_diff();
    const double* data_vec = cpu_data();
    for (int i = 0; i < count_; ++i) {
        proto->add_double_data(data_vec[i]);
    }
    if (write_diff) {
        const double* diff_vec = cpu_diff();
        for (int  i = 0; i < count_; ++i) {
            proto->add_double_data(diff_vec[i]);
        }
    }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;
} // namespace caffe


