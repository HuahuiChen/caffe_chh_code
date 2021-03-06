#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_ 

#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility> // pair

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Net {
  public:
    explicit Net(const NetParameter& param, const Net* root_net = NULL);
    explicit Net(const string& param_file, Phase phase, const Net* root_net = NULL);
    virtual ~Net() {

    }

    void Init(const NetParameter& param);

    const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);

    const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
        LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
            << "will be removed in a future version. Use Forward().";
        return Forward(loss);    
    }

    Dtype ForwardFromTo(int start, int end);
    Dtype ForwardFrom(int start);
    Dtype ForwardTo(int end);

    const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>*> & bottom,
        Dtype* loss = NULL);
    void ClearParamDiffs();

    void Backward();
    void BackwardFromTo(int start, int end);
    void BackwardFrom(int start);
    void BackwardTo(int end);

    void Reshape();

    Dtype ForwardBackward() {
        Dtype loss;
        Forward(&loss);
        Backward();
        return loss;
    }

    void Update();

    void ShareWeights();

    void ShareTrainedLayersWith(const Net* other);

    void CopyTrainedLayersFrom(const NetParameter& param);
    void CopyTrainedLayersFrom(const string trained_filename);
    void CopyTrainedLayersFromBinaryProto(const string trained_filename);
    void CopyTrainedLayersFromHDF5(const string trained_filename);

    void ToProto(NetParameter* param, bool write_diff = false) const;
    void ToHDF5(const string& filename , bool write_diff = false) const;

    inline const string& name() const {
        return name_;
    }
    inline const vector<string>& layer_names() {
        return later_names_;
    }
    inline const vector<string>& blob_names() {
        return blob_names_;
    }
    inline const vector<shared_ptr<Blob<Dtype> > > & blobs() {
        return blobs_;
    }
    inline const vector<shared_ptr<Layer<Dtype> > > & layers() const {
        return layers_;
    }
    inline Phase phase() const {
        return phase_;
    }
    inline const vector<vector<Blob<Dtype>* > >& bottom_vecs() const {
        return bottom_vecs_;
    }
    inline const vector<vetor<Blob<Dtype>* > >& top_vecs() const {
        return top_vecs_;
    }
    inline const vector<int>& top_ids(int i) const {
        CHECK_GE(i, 0) << "Invalid layer id.";
        CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id.";
        return top_id_vecs_[i];
    }
    inline const vector<int>& bottom_ids(int i) const {
        CHECK_GE(i, 0) << "Invalid layer id.";
        CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id.";
        return bottom_id_vecs_[i];
    }
    inline const vector<vector<bool>> bottom_need_backward() const {
        return bottom_need_backward_;
    }
    // ??
    inline const vector<Dtype>& blob_loss_weights() const {
        return blob_loss_weights_;
    }
    inline const vector<bool>& layer_need_backward() const {
        return layer_need_backward_;
    }
    inline const vector<shared_ptr<Blob<Dtype> >& params() const {
        return params_;
    }
    inline const vector<Blob<Dtype>*>& learnable_params() const {
        return learnable_params_;
    }
    // returns the learnable parameter learning rate multipliers
    inline const vector<float>& param_lr() const {
        return params_lr_;
    } 
    inline const vector<bool>& has_params_lr() const {
        return has_params_lr_;
    }
    // returns the learnable parameter decay multipliers.
    inline const vector<float>& params_weight_decay() const {
        return params_weight_decay_;
    }
    inline const vector<bool>& has_params_decay() const {
        return has_params_dacay_;
    }
    const map<string, int>& param_names_index() const {
        return param_names_index_;
    }
    inline const vector<int>& param_owners() const {
        return param_owners_;
    }
    inline const vector<string>& param_display_names() const {
        return param_display_names_;
    }

    inline int num_inputs() const {
        return net_input_blobs_.size();
    }
    inline int num_outputs() const {
        return net_output_blobs_.size();
    }
    inline const vector<Blob<Dtype>*>& input_blobs() const {
        return net_input_blobs_;
    }
    inline const vector<Blob<Dtype>*>& output_blobs() const {
        return net_output_blobs_;
    }
    inline const vector<int>& input_blob_indices() const {
        return net_input_blob_indices_;
    }
    inline const vector<int>& output_blob_indices() const {
        return net_output_blob_indices_;
    }
    bool has_blob(const string& blob_name) const;
    const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
    bool has_layer(const string& layer_name) const;
    const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

    void set_debug_info(const bool value) {
        debug_info_ = value;

    }

    static void FilterNet(const NetPrameter& param, NetPrameter* param_filtered);
    static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
        const string& layer_name);

  protected:
    // Append a new top blob to the net.
    void AppendTop(const NetParameter& param, const int layer_id,
        const int top_id, set<string>* available_blobs,
        map<string, int>* blob_name_to_idx);  

    int AppendBottom(const NetPrameter& param, const int layer_id, 
        const int bottom_id, set<string>* available_blobs,
        map<string, int>* blob_name_to_idx);

    void AppendParam(const NetParameter& param, const int layer_id, 
        const int param_id);

    /// @brief Helper for displaying debug info in Forward.
    void ForwardDebugInfo(const int layer_id);
    void BackwardDebugInfo(const int layer_id);
    void UpdateDebugInfo(const int param_id);

    string name_;
    /// @brief The phase: TRAIN or TEST
    Phase phase_;
    vector<shared_ptr<Layer<Dtype> > > layers_;
    vector<string> layer_names_;
    vector<string, int> layer_names_index_;
    vector<bool> layer_need_backward_;

    /// @brief The Blob stores intermedia results between the layer.
    vector<shared_ptr<Blob<Dtype> > > blobs_;
    vector<string> blob_names_;
    map<string, int> blob_names_index_;
    vector<bool> blob_need_backward_;
    
    /// bottom_vecs stores the vectors containing the input for each layer.
    /// They don't actually host the blobs (blobs_ dose), so we simply store 
    /// pointers.
    vector<vector<Blob<Dtype>*> > bottom_vecs_;
    vector<vector<int> > bottom_id_vecs_;
    vector<vector<bool> > bottom_need_backward_;

    vector<vector<Blob<Dtype>* > > top_vecs_;
    vector<vector<int> > top_id_vecs_;
    vector<vector<bool> > top_need_backward_;

    /// Vector of weights in the loss function of each net blob,
    /// indexed by blob_id.
    vector<Dtype> blob_loss_weights_; //??
    
    vector<vector<int> > param_id_vecs_;
    vector<int> param_owners_; // for share data
    vector<string> param_display_names_;
    vector<pair<int, int> > param_layer_indices_;
    map<string, int> param_names_index_;

    vector<int> net_input_blob_indices_;
    vector<int> net_output_blob_indices_;
    vector<Blob<Dtype>*> net_input_blobs_;
    vector<Blob<Dtype>*> net_output_blobs_;
    
    /// The parameters in the Net.
    vector<shared_ptr<Blob<Dtype> > > params_;
    vector<Blob<Dtype>*> learnable_params_;

    /**
     * The mapping from params_ -> learnable_params_: we have 
     * learnable_params_ids_.size() == param_.size(),
     * and learnable_params_[learnable_params_ids_[i]] == param_[i].get()
     * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer 
     * and learnable_params_[learnable_params_ids_[i]] gives its owner.
     */
    vector<int> learnable_params_ids_;
    /// The learning rate multipliers for learnable_params_;
    vector<float> params_lr_;
    vector<bool> has_params_lr_;

    /// The weight decay multipliers for learnable_params_ 
    vector<float> params_weight_decay_;
    vector<bool> has_params_dacay_;

    size_t memory_used_;
    bool debug_info_;

    const Net* const root_net_;

    DISABLE_COPY_AND_ASSIGN(Net);

    

 


};


} // namespace caffe


#endif
