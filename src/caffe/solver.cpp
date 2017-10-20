#include <cstdio>
#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
    action_request_function_ = func;
}

SolverAction::Enum GetRequestedAction() {
    if (action_request_function_) {
        return action_request_function_();
    }
    return SolverAction::NONE;
}

template<typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false)  {
    Init(param);
}

template<typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false)  {
    SolverParameter param;
    ReadSolverParamsFromTextFileOrDie(param_file, param);
    Init(param);
}

template<typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
    CHECK_EQ(Caffe::root_solver() || root_solver_)
        << "root_solver_ needs to be set for all non-root solvers";
    LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
        << std::endl << param.DebugString();
    param_ = param;
    CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
    CheckSnapshotWritePermissions();
    if (Caffe::root_solver() && param_.random_seed() >= 0) {
        Caffe::set_random_seed(param_.random_seed());
    }
    InitTrainNet();
    if (Caffe::root_solver()) {
        InitTestNets();
        LOG(INFO) << "Solver scaffolding done.";
    }
    iter_ = 0;
    current_step_ = 0;
}


template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
    const int num_train_nets = param_.has_net() + param.has_net_param() + 
        param_.has_trian_net() + param_.has_train_net_param();
    const string& field_names = "net, net_param, trian_net, train_net_param";
    CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
        << "using one of these fields: " << field_names;
    CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
        << "one of these fields specifying a train_net: " << field_names;
    NetParameter net_param;
    if (param_.has_train_net_param()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "Creating training net specified in train_net_param.";
        net_param.CopyFrom(param_.train_net_param());    
    } else if (param_.has_train_net()) {
        LOG_IF(INFO, caffe::root_solver())
            << "Creating training net from trian_net file: " << param.train_net();
        ReadNetParamsFromTextFileOrDie(param_.trian_net(), &net_param);    
    }            
    if (param_.has_net_param()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "Creating training net specified in net_param.";
        net_param.CopyFrom(param_.net_param());    
    }
    if (param_.has_net()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "Creating training net from net file: " << param_.net();
        ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);    
    }

    NetState net_state;
    net_state.set_phase(TRAIN);
    net_state.MergeFrom(net_param.state());
    net_state.MergeFrom(param_.train_state());
    net_param.mutable_state()->CopyFrom(net_state);

    // Create Net 
    if (Caffe::root_solver()) {
        net_.reset(new Net<Dtype>(net_param));
    } else {
        net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
    }

}

// TODO
template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
    CHECK(Caffe::root_solver());
    const bool has_net_param = param_.has_net_param();
    const bool has_net_file = param_.has_net();
    const int num_generic_nets = has_net_param + has_net_file;

}

// Key function 
template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
    const int start_iter = iter_;
    const int stop_iter = iter_ + iters;
    int average_loss = this->param.average_loss();
    losses_.clear();
    smoothed_loss_ = 0;
    while(iter_ < stop_iter) {
        net_->ClearParamDiffs();
        if (param_.test_interval() && iter_ % param_.test_interval() == 0
            && (iter_ > 0 || param_.test_initialzation())
            && Caffe::root_solver()) {
            TestAll();
            if (requested_early_exit_) {
                // Break out of the while loop because stop was requested while testing.
                break;
            }

        }
        for (int i = 0; i < callbacks_.size(); ++i)  {
            callbacks_[i]->on_start();
        }
        const bool display = param_.display() && iter_ % param_.display() == 0;
        net_->set_debug_info(display && param_.debug_info());

        Dtype loss = 0;
        for (int i = 0; i < param_.iter_size(); ++i) {
            loss += net_->ForwardBackward();
        }

        loss /= param_.iter_size();

        UpdateSmoothedLoss(loss, start_iter, average_loss);
        if (display) {
            LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_ 
                << ", loss = " << smoothed_loss_;
            const vector<Blob<Dtype>* >& result = net_->output_blobs();
            int score_index = 0;
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                const string& output_name = 
                    net_->blob_names()[net_->output_blob_indices()[j]];
                const Dtype* loss_weight = 
                    net_->blob_loss_weights()[net_->output_blob_indices()[j]];
                for (int k = 0; k < result[j]->count; ++k) {
                    ostringstream loss_msg_stream;
                    if (loss_weight) {
                        loss_msg_stream << " (* " << loss_weight 
                            << " = " << loss_weight * result_vec[k] << " loss)";
                    }
                    LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
                        << score_index++ << ": " << output_name << " = "
                        << result_vec[k] << loss_msg_stream.str();
                }        
            }    
        }

        for (int i = 0; i < callbacks_.size(); ++i) {
            callbacks_[i]->on_gradients_ready();
        }
        ApplyUpdate();
        ++iter_;

        SolverAction::Enum request = GetRequestedAction();

        if ((param_.snapshot()
            && iter_ % param_.snapshot() == 0
            && Caffe::root_solver()) ||
            (request == SolverAction::SNAPSHOT)) {
            Snashot();
        }
        if(SolverAction::STOP == request) {
            requested_early_exit_ = true;
            break;
        }
    }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Solving " << net_->name();
    LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

    requested_early_exit_ = false;

    if (resume_file) {
        LOG(INFO) << "Restoring previous solver status from " << resume_file;
        Restore(resume_file);
    }

    int start_iter = iter_;
    Step(param_.max_iter() - iter_);

    if (param_.snapshot_after_train()
        && (!param_.shapshot() || iter_ % param_.snapshot() != 0)) {
        Snapshot();
    }
    if (requested_early_exit_) {
        LOG(INFO) << "Optimizing stopped early.";
        return;
    }

    if (param_.display() && iter_ % param_.display() == 0) {
        int average_loss = this->param_.average_loss();
        Dtype loss;
        net_->Forward(&loss);

        UpdateSmoothedLoss(loss, start_iter, average_loss);
        LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
        TestAll();
    }
    LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
    for (int test_net_id = 0; test_net_id < test_nets_.size() && !requested_early_exit_;
        ++test_net_id) {
        Test(test_net_id);
    }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
    CHECK(Caffe::root_solver());
    LOG(INFO) << "Iteration " << iter_ 
        << ", Test net (#" << test_net_id << ")";
    CHECK_NOTNULL(test_nets_[test_net_id].get())->ShareTrainedLayersWith(net_.get());
    vector<Dtype> test_score;
    vecotr<int> test_score_output_id;
    const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
    Dtype loss = 0;
    for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
        SolverAction::Enum request = GetRequestedAction();
        while (request != SolverAction::NONE) {
            if (SolverAction::SNAPSHOT == request) {
                Snapshot();
            } else if (SolverAction::STOP == request) {
                requested_early_exit_ = true;1
            }
            request = GetRequestedAction();
        }
        if (requested_early_exit_) {
            break;
        }

        Dtype iter_loss;
        const vector<Blob<Dtype>*>& result = 
            test_net->Forward(&iter_loss);
        if (param_.test_compute_loss()) {
            loss += iter_loss;
        }    
        if (i == 0) {
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                for (int k = 0; k < result[j]->count(); ++k) {
                    test_score.push_back(result_vec[k]);
                    test_score_output_id.push_back(j);
                }
            }
        } else {
            int idx = 0;
            for (int j = 0; j < result.size(); ++j) {
                const Dtype* result_vec = result[j]->cpu_data();
                for (int k = 0; k < result[j]->count(); ++k) {
                    test_score[idx++] += result_vec[k];
                }
            }
        }

    }    
    if (requested_early_exit_) {
        LOG(INFO) << "Test interrupted.";
        return;
    }
    if (param_.test_compute_loss()) {
        loss /= param_.test_iter(test_net_id);
        LOG(INFO) << "Test loss: " << loss;
    }

    for (int i = 0; i < test_score.size(); ++i) {
        const int output_blob_index = 
            test_net->output_blob_indices()[test_score_output_id[i]];
        const string& output_name = test_net->blob_names()[output_blob_index];    
        const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
        ostringstream loss_msg_stream;
        const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight 
                << " = " << loss_weight * mean_score << " loss)";
        }
        LOG(INFO) << "  Test net output #" << i << ": " << output_name << " = "
            << mean_score << loss_msg_stream.str();
    }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
    CHECK(Caffe::root_sovler());
    string model_filename;
    switch (param_.snapshot_format()) {
        case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
            model_filename = SnapshotToBinaryProto();
            break;
        case caffe::SolverParameter_SnapshotFormat_HDF5:
            model_filename = SnapshotToHDF5();
            break;
        default:
            LOG(FATAL) << "Unsupported snapshot format.";        
    }
    SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
    if (Caffe::root_solver() && param_.snapshot()) {
        CHECK(param_.has_snapshot_prefix())
            << "in solver params, snapshot is specified but snapshot_prefix is not";
        string probe_filename = SnapshotFilename(".tempfile");
        std::ofstream probe_ofs(probe_filename.c_str());
        if (probe_ofs.good()) {
            probe_ofs.close();
            std::remove(probe_filename.c_str());
        } else {
            LOG(FATAL) << "Cannot write to snapshot prefix '"
                << param_.snapshot_prefix() << "'. Make sure "
                << " that the directory exists and is writeable.";
        }    
            
    }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string& extension) {
    return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
        + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
    string model_filename = SnapshotFilename(".caffemodel");
    LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
    NetParameter net_param;
    net_->ToProto(&net_param, param_.snapshot_diff());
    WriteProtoToBinaryFile(net_param, model_filename);
    return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
    string model_filename = SnapshotFilename(".caffemodel");
    LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
    net_->ToHDF5(model_filename, param_.snapshot_diff());
    return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
    CHECK(Caffe::root_solver());
    string state_filename(state_file);
    if (state_filename.size() >= 3 &&
        state_filename.compare(state_filename - 3, 3, ".h5") == ) {
        RestoreSolverStateFromHDF5(state_filename);
    } else {
        RestoreSolverStateFromBinaryProto(state_filename);
    }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
    if (losses_.size() < average_loss) {
        losses_.push_back(loss);
        int size = losses_.size();
        smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
    } else {
        int idx = (iter_ - start_iter) % average_loss;
        smoothed_loss_ += (loss - losses_[idx]) / average_loss;
        losses_[idx] = loss;
    }
}

INSTANTIATE_CLASS(Solver);

} // namespace caffe 

