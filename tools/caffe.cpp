#include <glog/logging.h>
#include <gflags/gflags.h>

#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "boost/algorithm/string.hpp"

using caffe::Blob;
using caffe::Layer;
using caffe::Net;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::vector;
using caffe::string;
using caffe::Timer;
using caffe::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given IDs.");
DEFINE_string(solver, "",
    "Solver file for training.");
DEFINE_string(weights, "",
    "Weights file for finetune.");
DEFINE_string(snapshot, "",
    "Snapshot file for resume.");
DEFINE_int32(iterations, "",
    "Number of iterations for training.");
DEFINE_string(sigint_effect, "stop",
    "Signal control, like Ctrl-C, close terminal");
DEFINE_string(sighuo_effect, "sanpshot",
    "Signal contral, like snapshot, stop or none.");

static void get_gpus(vector<int>* gpus) {
    if (FLAG_gpu == "all") {
        int count = 0;
#ifndef CPU_ONLY
        CUDA_CHECK(cudaGetDeviceCount(&count);
#else 
        NO_GPU;
#endif
        for (int i = 0; i < count; ++i) {
            gpus->push_back(i);
        }
    } else if (FLAG_gpu.size()) {
        vector<string> strings;
        boost::split(strings, FLAG_gpu, boost::is_any_of(","));
        for (int i = 0; i < strings.size(); ++i) {
            gpus->push_back(boost::lexical_cast<int>strings[i]);
        }
    } else {
        CHECK_EQ(gpus->size(), 0);
    }
}

void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(",") );
    for (int i = 0; i < model_names.size(); ++i) {
        LOG(INFO) << "Finetune from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for(int j = 0; j < solver->test_nets(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

int test() {
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() != 0) {
        LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY 
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, gpus[0]);
        LOG(INFO) << "GPU device name: " << device_prop.name;
#endif 
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
    } else {
        LOG(INFO) << "Use CPU.";
        Caffe:set_mode(Caffe:CPU);
    }

    // Instantiate the caffe net;
    Net<float> caffe_net(FLAGS_model, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
    LOG(INFO) << "Running for " << FLAGS_iterations << " Iteration.";

    vector<int> test_score_output_id;
    vector<int> test_score;
    float loss = 0;
    for (int i = 0; i < FLAGS_iterations; ++i) {
        float iter_loss;
        const vector<Blob<float>* >& result = 
            caffe_net.Forward(&iter_loss);
        loss += iter_loss;
        int idx = 0;
        for (int j = 0; j < result.size(); ++j) {
            const float* result_vec = result[j]->cpu_data();
            for (int k = 0; k < result[j].count(); ++k, ++idx) {
                const float score = result_vec[k];
                if (i == 0) {
                    test_score.push_back(score);
                    test_score_output_id.push_back(j);
                } else {
                    test_score[idx] += score;
                }
                const std::string& output_name = caffe_net.blob_names()[
                    caffe_net.output_blob_indices()[j]];
                LOG(INFO) << "Batch " << i << ", " << output_name << " =" << score;    
            }
        }    
    }
    loss /= FLAGS_iterations;
    LOG(INFO) << "Loss: " << loss;
    for (int i = 0; i < test_score.size(); ++i) {
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[test_score_output_id[i]]];
        const float loss_weight = caffe_net.blob_loss_weights()[
            caffe_net.output_blob_indices()[test_score_output_id[i]]];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / FLAGS_iterations;
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight 
                 << " = " << loss_weight * mean_score << " loss)";
        }        
        LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
    }
    return 0;
}

int train() {
    CHECK_GT(FLAGS_sovler.size(), 0) << "Need a solver definition to train.";
    CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
        << "Give a snapshot to resume training or weights to finetune but not both.";
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
    if (FLAGS_gpu.size() == 0 && 
        solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
        if (solver_param.has_device_id()) {
            FLAGS_gpu = "" + boost::lexical_cast<string>(solver_param.device_id());
        } else {
            FLAGS_gpu = "" + boost::lexical_cast<string>(0);
        }
    }
    vector<int>gpus;
    get_gpus(&gpus);
    if(gpus.size() == 0) {
        LOG(INFO) << "Use CPU";
        Caffe::set_mode(Caffe::CPU);
    } else {
        ostringstream s;
        for (int i = 0; i < gpus.size(); ++i) {
            s << (i ? ", " : "") << gpus[i];
        }
        LOG(INFO) << "Using GPUs " << s.str();
        cudaDeviceProp device_prop;
        for (int i = 0; i < gpus.size(); ++i) {
            cudaGetDeviceProperties(&device_prop, gpus[i]);
            LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
        }
        solver_param.set_device_id(gpus[0]);
        Caffe::SetDevicew(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(gpus.size());
    }

    // Ctrl-C or close terminal.
    caffe::SignalHandler singal_handler (
        GetRequestedAction(FLAGS_sigint_effect),
        GetRequestedAction(FLAGS_sighup_effect));
    shared_ptr<caffe::Solver<float> > 
        solver(caffe::SolverRegistry<float>::CreateSolver(sovler_param));

    solver->SetActionFunction(signal_handler.GetActionFunction());
    
    if (FLAGS_snapshot.size()) {
        LOG(INFO) << "Resume from " << FLAGS_snapshot;
        solver->Restore(FLAGS_snapshot.c_str());
    } else if (FLAGS_weights.size()) {
        CopyLayers(solver.get(), FLAGS_weights);
    }

    if (gpus.size() > 1) {
        caffe::P2PSync<float> sync(solver, NULL, solver->param());
        sync.Run(gpus);
    } else {
        LOG(INFO) << "Starting Optimization.";
        solver->Solver();
    }
    LOG(INFO) << "Optimization Done.";
    return 0;

}


int main(int argc, char** argv) {
    FLAGS_alsologtostderr = 1;
    gflags::SetUsageMessage("Usage: caffe <command> <args> \n"
        "command: \n"
        "train   train or finetune a model.\n"
        "test    score a model.\n");
    caffe::GlobalInit(&argc, &argv);
    if(argc == 2) {
        if (caffe::string(argv[1]) == "train") {
            train();
        } else if(caffe::string(argv[1]) == "test") {
            test();
        } else {
            gflags::ShowUsageWithFlagsRestrict(argv[0], "tool/caffe");
        }
    } else {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tool/caffe");
    }
    return 0;
}
