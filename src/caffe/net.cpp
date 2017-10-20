#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/test/test_caffe_main.hpp"


namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, const Net* root_net)
    : root_net_(root_net_) {
        Init(param);
    }    
template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase, const Net* root_net)
    : root_net_(root_net_) {
        NetParameter param;
        ReadNetParamsFromTextFileOrDie(param_file, &param);
        param.mutable_state()->set_phase(phase);
        Init(param);
    }    

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
    CHECK(Caffe::root_solver() || root_net_)
        << "root_net_ needs to be set for all non-root solvers";
    phase_ = in_param.state().phase();
    NetParameter filtered_param;
    FilterNet(in_param, &filter_param);
    LOG_IF(INFO, Caffe::root_solver())
        << "Initializing net from parameters: " << std::endl;
        << filtered_param.DebugString();
    /// Create a copy of filtered_param with splits added where necessary.
    NetParameter param;
    InsertSplits(filtered_param, &param);
    /// Build all layers and connections.
    name_ = param.name();
    map<string, int> blob_name_to_idx;
    set<string> available_blobs;
    memory_used_ = 0;
    // For each layer, set up its input and output
    bottom_vecs_.resize(param.layer_size());
    top_vecs_.resize(param.layer_size());
    bottom_id_vecs_.resize(param.layer_size());
    param_id_vecs_.resize(param.layer_size());
    top_id_vecs_.resize(param.layer_size());
    bottom_need_backward_.resize(param.layer_size());
    for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
        /// For non-root solvers, whether this layer is shared from root_net_.
        bool shared_from_root = !Caffe::root_solver()
            && root_net_->layers_[layer_id]->ShareInParallel();
        if (!param.layer(layer_id).has_phase()) {
            param.mutable_layer(layer_id)->set_phase(phase_);
        }    
        // Setup layer.
        const LayerParameter& layer_param = param.layer(layer_id);
        if (layer_param.propagate_down_size() > 0) {
            CHECK_EQ(layer_param.propagrate_down_size(),
                layer_param.bottom_size())
                << "propagate_down param must be specified "
                << "either 0 or bottom_size times ";
        }
        if (share_from_root) {
            LOG(INFO) << "Sharing layer " << layer_param.name() << " from root net";
            layer_.push_back(root_net_->layers_[layer_id]);
            layer_[layer_id]->SetShared(true);
        } else {
            // Add layer.
            layers_.push_back(LayerRegistry<Dtype::CreateLayer(layer_param));
        }
        layer_names_.push_back(layer_param.name());
        LOG_IF(INFO, Caffe::root_solver())
            << "Creating layer " << layer_param.name();
        bool need_backward = false;
        
        for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
            ++bottom_id) {
            const int blob_id = AppendBottom(param, layer_id, bottom_id,
                &available_blobs, &blob_name_to_idx);
            need_backward |= blob_need_backward_[blob_id];
        }    
        int num_top = layer_param.top_size();
        for (int top_id = 0; top_id < num_top; ++top_id) {
            AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
            if (layer_param.type() == "Input") {
                const int blob_id = blobs_.size() - 1;
                net_input_blob_indices_.push_back(blob_id);
                net_input_blobs_.push_back(blobs_[blob_id].get());
            }
        }
        // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter 
        // specified fewer than the required number (as specified by 
        // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
        Layer<Dtype>* layer = layers_[layer_id].get();
        if (layer->AutoTopBlobs()) {
            const int needed_num_top = 
                std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());
            for (; num_top < needed_num_top; ++num_top) {
                AppendTop(param, layer_id, num_top, NULL, NULL);
            }    
        }
        if (share_from_root) {
            const vector<Blob<Dtype>*>& base_top = root_net_->top_vecs_[layer_id];
            const vector<Blob<Dtype>*>& this_top = this->top_vecs_[layer_id];
            for (int top_id = 0; top_id < base_top.size(); ++top_id) {
                this_top[top_id]->ReshapeLike(*base_top[top_id]);
                LOG(INFO) << "Created top blob " << top_id << " (shape: "
                    << this_top[top_id]->shape_string() << ") for shared layer "
                    << layer_param.name();
            }
        } else {
            layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
        }
        LOG_IF(INFO, Caffe::root_solver())
            << "Setting up " << layer_names_[layer_id];
        for (int top_id = 0;  top_id < top_vecs_[layer_id].size(); ++top_id) {
            if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
                blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, 
                    Dtype(0))
            }
            blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
            LOG_IF(INFO, Caffe::root_solver())
                << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
            if (layer->loss(top_id)) {
                LOG_IF(INFO, Caffe::root_solver())
                    << "      with loss weight " << layer->loss(top_id);
            }    
            memory_used_ += top_vecs_[layer_id][top_id]->count();
         }
         LOG_IF(INFO, Caffe::root_solver())
            << "Memory required for data: " << memory_used_ * sizeof(Dtype);
         const int param_size = layer_param.param.size();
         const int num_param_blobs = layer_[layer_id]->blobs().size();
         CHECK_LE(param_size, num_param_blobs)
            << "Too many params specified for layer " << layer_param.name();
         ParamSpec default_param_spec;
         for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
             const ParamSpec* param_spec = (param_id < param_size) ? 
                &layer_param.param(param_id) : &default_param_spec;
             cosnt bool param_need_backward = param_spec->lr_mult() !0;
             need_backward |= param_need_backward;
             layer_[layer_id]->set_param_propagate_down(param_id,
                param_need_backward);   
         }      
         for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
             AppendParam(param, layer_id, param_id);
         }
         layer_need_backward_.push_back(need_backward);
         if (need_backward) {
             for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
                 blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
             }
         }
    }        
    // Go throut the net backwards to determine which blobs contribute to the 
    // loss. We can skip backward computation for blobs that don't contribute 
    // to the loss.
    // Also checks if all bottom blobs don't need backward computation 
    // (possible because the skip_propagate_down param) and so we can skip 
    // backward computation for the entire layer
    set<string> blobs_under_loss;
    set<string> blobs_skip_backp;
    for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
        bool layer_contributes_loss = false;
        bool layer_skip_propagate_down = true;
        for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
            const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
            if (layers_[layer_id]->loss(top_id) || 
                (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
                layer_contributes_loss = true;
            }
            if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
                layer_skip_propagate_down = false;
            }
            if (layer_contributes_loss && !layer_skip_propagate_down)
                break;
        }
        if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
            layer_need_backward_[layer_id] = false;
            for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
                ++bottom_id;) {
                bottom_need_backward_[layer_id][bottom_id] = false;
            }
        }
        if (!layer_contributes_loss) {
            layer_need_backward_[layer_id] = false;
        }
        if (Caffe::root_solver()) {
            if (layer_need_backward_[layer_id]) {
                LOG(INFO) << layer_names_[layer_id] << "need backward computation.";
            } else {
                LOG(INFO) << layer_names_[layer_id]
                    << " does not need backward computation.";
            }
        }
        for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
            ++bottom_id;) {
            if (layer_contributes_loss) {
                const string& blob_name = 
                    blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
                blobs_under_loss.insert(blob_name);;    
            } else {
                bottom_need_backward_[layer_id][bottom_id] = false;
            }
            if(!bottom_need_backward_[layer_id][bottom_id]) {
                    const string& blob_name = 
                        blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
                    blobs_skip_backp.insert(blob_name);    
            }
        }
    }

    // Force_backward if needed.
    if (param.force_backward()) {
        for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
            layer_need_backward_[layer_id] = true;
            for (int bottom_id = 0;
                bottom_id < bottom_need_backward_[layer_id].size();
                ++bottom_id) {
                bottom_need_backward_[layer_id][bottom_id] = 
                    bottom_need_backward_[layer_id][bottom_id] ||
                    layers_[layer_id]->AllowForceBackward(bottom_id);
                blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
                    blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
                    bottom_need_backward_[layer_id][bottom_id];    
            }
            for (int param_id = 0; param_id < layers_[layer_id]->blobs().size()
                ++param_id) {
                layers_[layer_id]->set_param_propagate_down(param_id, true);
            }
        }
    }

    for (set<string>::iterator it = available_blobs.begin();
        it != available_blobs.end(); ++it) {
        LOG_IF(INFO, Caffe::root_solver())
            << "This network produces output " << *it;
        net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
        net_output_blob_indices_.push_back(blob_name_to_idx[*it]);    
    }
    for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
        blob_names_index_[blob_names_[blob_id]] = blob_id;
    }
    for (size_t layer_id = 0; layer_id < layer_names.size(); ++layer_id) {
        layer_names_index_[layer_names_[layer_id]] = layer_id;
    }
    ShareWeights();
    debug_info_ = param.debug_info();
    LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
void Net<Dtype>::FilterNer(const NetParameter& param,
    NetParameter* param_filtered) {
    NetState net_state(param.state());
    param_filtered->CopyFrom(param);
    param_filtered->clear_layer();
    for (int i = 0; i < param.layer_size(); ++i) {
        const LayerParameter& layer_param = param.layer(i);
        const string& layer_name = layer_name.name();
        CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
            << "Specify either include rules or exclude rules; not both.";
        bool layer_included = (layer_param.include_size() == 0);
        for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
            if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
                layer_included = false;
            }
        }    
        for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
            if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
                layer_included = true;
            }
        } 
        if (layer_included) {
            param_filtered->add_layer()->CopyFrom(layer_param);
        }
    }
}


template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
    if (rule.has_phase()) {
        if (rule.phase() != state.phase()) {
            LOG_IF(INFO, Caffe::root_solver())
                <<  "The NetState phase (" << state.phase()
                << ") differed from the phase (" << rule.phase()
                << ") specified by a rule in layer " << layer_name;
            return false;
        }
    }
// TODO

}


template <typename Dtype>
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
    const int top_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
    shared_ptr<LayerParameter> layer_param( 
        new LayerParameter(param.layer(layer_id)));
    const string& blob_name = (layer_param->top_size() > top_id) ?
        layer_param->top(top_id) : "(automatic)";
    if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
            blob_name == layer_param->bottom(top_id)) {
        // In-place computation 
        LOG_IF(INFO, Caffe::root_solver())
            << layer_param->name() << " -> " << blob_name << " (in-place)";
        top_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]].get());
        top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);    
    } else if (blob_name_to_idx &&
        blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
        LOG(FATAL << "Top blob '" << blob_name 
            << "produced by multiple sources.";
    }  else {
        // Normal output. 
        if (Caffe::root_solver()) {
            LOG(INFO) << layer_param->name() << " -> " << blob_name;
        }
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        const int blob_id = blobs_.size();
        blobs_.push_back(blob_pointer);
        blob_names_.push_back(blob_name);
        blob_need_backward_.push_back(false);
        if (blob_name_to_idx) {
            (*blob_name_to_idx)[blob_name] = blob_id;
        }
        top_id_vecs_[layer_id].push_back(blob_id);
        top_vecs_[layer_id].push_back(blob_pointer.get());
    } 
    if (available_blobs) {
        available_blobs->insert(blob_name);
    }
}

template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
    const LayerParameter& layer_param = param.layer(layer_id);
    const string& blob_name = layer_param.bottom(bottom_id);
    if (available_blobs->find(blob_name) == available_blobs->end()) {
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
            << layer_param.name() << "', bottom index " << bottom_id << ")";
    }
    const int blob_id = (*blob_name_to_idx)[blob_name];
    LOG_IF(INFO, Caffe::root_solver())
        << layer_names_[layer_id] << " <- " << blob_name;
    bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
    bottom_id_vecs_[layer_id].push_back(blob_id);
    available_blobs->erase(blob_name);
    bool propagate_down = true;
    if (layer_param.propagate_down_size() > 0)
        propagate_down = layer_param.propagate_down(bottom_id);
    const bool need_backward = blob_need_backward_[blob_id] && propagate_down;
    bottom_need_backward_[layer_id].push_back(need_backward);
    return blob_id;        

}

template <typename Dtype>
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
    const int param_id) {
    const LayerParameter& layer_parm = layers_[layer_id]->layer_param();
    const int param_size = layer_param.param_size();
    string param_name = 
        (param_size > param_id) ? layer_param.param(param_id).name() : "";
    if (param_name.size()) {
        param_display_names_.push_back(param_name);
    } else {
        ostringstream param_display_names;
        // TODO
    }    
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
    CHECK_GE(start, 0);
    CHECK_LE(end, layers_.size());
    Dtype loss = 0;
    for (int i = start; i <= end; ++i) {
        Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
        loss += layer_loss;
        if (debug_info_) {
            ForwardDebugInfo(i);
        }
    }
    return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
    return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
    return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
    if (loss != NULL) {
        *loss = ForwardFromTo(0, layers_.size() - 1);
    } else {
        ForwardFromTo(0, layers_.size() - 1);
    }
    return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, Dtype* loss) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bootm, loss) "
        << "will be removed in a future version. Use Forward(loss).";
    for (int i = 0; i < bottom.size(); ++i) {
        net_input_blobs_[i]->CopyFrom(*bottom[i]);
    }    
    return Foward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
    CHECK_GE(end, 0);
    CHECK_LT(start, layers_.size());
    for (int i = start; i >= end; --i) {
        if (layer_need_backward_[i]) {
            layers_[i]->Backward(
                top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
            if (debug_info_) {
                BackwardDebugInfo(i);
            }    
        }
    }
}

// TODO 
template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {

}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {

}


template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int layer_id) {

}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
    int num_source_layers = other->layers().size();
    for (int i = 0; i < num_sorN; ++i) {
        Layer<Dtype>* source_layer = other->layers()[i].get();
        const string& source_layer_name = other->layer_names()[i];
        int target_layer_id = 0;
        while (target_layer_id != layer_names_.size() &&
            layer_names_[target_layer_id] != source_layer_name) {
            ++target_layer_id;
        }
        // TODO
    }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
    BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
    BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
    BackwardFromTo(layers_.size() - 1, 0);
    if (debug_info_) {
        Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
        for (int i = 0; i < learnable_params_.size(); ++i) {
            asum_data += learnable_params_[i]->asum_data();
            asum_diff += learnable_params_[i]->asum_diff();
            sumsq_data += learnable_params_[i]->sumsq_data();
            sumsq_diff += learnable_params_[i]->sumsq_diff();
        }
        const Dtype l2norm_data = std::sqrt(sumsq_data);
        const Dtype l2norm_diff = std::sqrt(sumsq_diff);
        LOG(ERROR) << "   [Backward[ All net params (data, diff): "
            << "L1 norm  = (" << asum_data << ", " << asum_diff << ");"
            << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
    }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
    for (int i = 0; i < layers_.size(); ++i) {
        layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
    }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
    int num_source_layers = param.layer_size();
    for (int i = 0; i < num_source_layers; ++i) {
        const LayerParameter& source_layer = param.layer(i);
        const string& source_layer_name = source_layer.name();
        int target_layer_id = 0;
        while (target_layer_id != layer_names_.size() &&
            layer_names_[target_layer_id] != source_layer_name) {
            ++target_layer_id;
        }
        if (target_layer_id == layer_names_.size()) {
           LOG(INFO) << "Ignoring source layer " << source_layer_name;
           continue; 
        }
        DLOG(INFO) << "Copying source layer " << source_layer_name;
        vector<shared_ptr<Blob<Dtype> > >& target_blobs = 
            layers_[target_layer_id]->blobs();
        CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
            << "Incompatible number of blobs for layer " << source_layer_name;
        for (int j = 0; j < target_blobs.size(); ++j) {
            if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
                Blob<Dtype> source_blob;
                const bool kReshape = true;
                source_blob.FromProto(source_layer.blobs(j), kReshape);
                LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
                    << source_layer_name << "'; shape mismatch. Source param shape is "
                    << source_blob.shape_string() << "; target param shape is "
                    << target_blobs[j]->shape_string() << ". "
                    << "To learn this layer's parameters form scratch rather than "
                    << "copying from  a saved net, rename the layer.";
            }
            const bool kReshape = false;
            target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
        }    

    }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
    if (trained_filename.size() >= 3 &&
            trained_filename.compare(trained_filename.size() - 3, 3, ".h5") == 0 ) {
        CopyTrainedLayersFromHDF5(trained_filename);
    } else {
        CopyTrainedLayersFromBinaryProto(trianed_filename);
    }
}


template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
    NetParameter param;
    ReadNetParamsFromTextFileOrDie(trained_filename, &param);
    CopyTrainedLayersFrom(param);
}

// TODO
template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string& trained_filename) {

}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
    param->Clear();
    param->set_name(name_);
    DLOG(INFO) << "Serializing " << layers_.size() << " layers";
    for (int i = 0; i < layers_.size(); ++i) {
        LayerParameter* layer_param = param->add_layer();
        layers_[i]->ToProto(layer_param, write_diff);
    }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
    hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(file_fid, 0)
        << "Couldn't open " << filename << " to save weights.";
    hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
    hid_t diff_hid = -1;
    if (write_diff) {
        diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT);
        CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
    }    
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
        const LayerParameter& layer_param = layers_[layer_id]->layer_param();
        string layer_name = layer_param.name();
        hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_GE(layer_data_hid, 0)
            << "Error saving weights to " << filename << ".";
        hid_t layer_diff_hid = -1;
        if (write_diff) {
            layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str()),
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            CHECK_GE(layer_diff_hid, 0)
                << "Error saving weights to " << filename << ".";
        }    
        int num_params = layers_[layer_id]->blobs().size();
        for (int param_id = 0; param_id < num_params; ++param_id) {
            ostringstream dataset_name;
            dataset_name << param_id;
            const int net_param_id = param_id_vecs_[layer_id][param_id];
            if (param_owners_[net_param_id] == -1) {
                hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
                    *params_[net_param_id]);
            }
            if (write_diff) {
                hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
                    *params_[net_param_id], true);
            }
        }
        H5Gclose(layer_data_hid);
        if (write_diff) {
            H5Gclose(layer_diff_hid);
        }
    }
    H5Gclose(data_hid);
    if (write_diff) {
        H5Gclose(diff_hid);
    }
    H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
    for (int i = 0; i < learnable_params_.size(); ++i) {
        learnable_params_[i]->Update();
    }
}

template <typename Dtype>
void Net<Dtype>::ClearParamsDiffs() {
    for (int i = 0; i < learnable_params_.size(); ++i) {
        Blob<Dtype>* blob = learnable_params_[i];
        switch (Caffe::mode()) {
            case Caffe::CPU:
                caffe_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_cpu_diff());
                break;
            case Caffe::GPU:
#ifndef CPU_ONLY
            caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_gpu_diff());                
#else 
            NO_GPU;
#endif
            break;
        }
    }
}

template <typename Dtype>
void Net<Dtype>::ShareWeights() {
    for (int i = 0; i < params_.size(); ++i) {
        if (param_owners_[i] < 0) {
            continue;
        }
        params_[i]->ShareData(*params_[param_owners_[i]]);
        params_[i]->ShareDiff(*params_[param_owners_[i]]);
    }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
    return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
        shared_ptr<Blob<Dtype> > blob_ptr;
        if (has_blob(blob_name)) {
            blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
        } else {
            blob_ptr.reset((Blob<Dtype>*)(NULL));
            LOG(WARNING) << "Unknown blob name " << blob_name;
        }
        return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const  {
    return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
    shared_ptr<Layer<Dtype> > layer_ptr;
    if (has_layer(layer_name)) {
        layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
    } else {
        layer_ptr.reset((Layer<Dtype>*)(NULL));
        LOG(WARNING) << "Unknown layer name " << layer_name;
    }
    return layer_ptr;

}
} // namespace caffe
