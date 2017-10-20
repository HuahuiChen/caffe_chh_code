/**
 * @brief A solver factory allows one to register solvers.
 * 
 *  SovlerRegister<Dtype>::CreateSolver(param);
 *  
 * Assuming that we have a solver like:
 *      template <typename Dtype>
 *      class MyAwesomeSolver : public Solver<Dtype> {
 *           //your implementations;
 *      }
 * and its type is its C++ class name, but without the "Solver" at the end 
 * ("MyAwesomeSolver"-> "MyAwesome").
 * 
 * There are two ways to register a solver:
 * 
 * 1. If the solver is going to be created simply by its constructor, in your c++
 *    file, add the following line:
 *    
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *    
 * 2. If the sovler is going to be created by another creator function, in the 
 *    format of:
 *    
 *    template <typename Dtype>
 *    Solver<Dtype* > GetMyAwesomeSolver (const SolverParameter& param) {
 *        // your implementations;
 *    } 
 *    
 *    then you can register the creator function like:
 *    
 *    REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver);
 *    
 *    Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_HPP_
#define CAFFE_SOLVER_FACTORY_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegisterer {
  public:
    typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
    typedef std::map<string, Creator> CreatorRegistry;
    
    static CreatorRegistry& Registry() {
        static CreatorRegistry* g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }    

    static void AddCreator(const string& type, Creator creator) {
        CreatorRegistry& registry = Registry();
        CHECK_EQ(registry.count(type), 0)
            << "Solver type " << type << "already registered.";
        registry[type] = creator;    
    }

    // Get a solver using a SolverParameter 
    static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
        const string& type = param.type();
        CreatorRegistry& registry = Registry();
        CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << 
            type << " (knows type: " << SolverTypeListString() << ")";
        return registry[type](param);    
    }

    static vector<string> SolverTypeList() {
        CreatorRegistry& registry = Registry();
        vector<string> solver_types;
        for (typename CreatorRegistry::iterator iter = registry.begin();
            iter != registry.end(); ++iter) {
            solver_types.push_back(iter->first);
        }
        return solver_types;
    }
  
  private:
    SolverRegistry() {

    }  

    static string SolverTypeListString() {
        vector<string> solver_types = SovlerTypeList();
        string solver_types_str;
        for (vector<string>::iterator iter = solver_types.begin();
             iter != solver_types.end(); ++iter ) {
            if (iter != solver_types.begin();) {
                solver_types_str += ", ";
            }
            solver_types_str += *iter;
        }
        return solver_types_str;
    }
};

template <typename Dtype>
class SolverRegisterer {
  public:
    SolverRegisterer(const string& type,
        Solver<Dtype>* (*creator)(const SolverParameter&)) {
        SolverRegistry<Dtype>::AddCreator(type, creator);
    }  
};


#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
    static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);  \
    static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>);

#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
    const SolverParameter& param)                                              \
    {                                                                          \
        return new type##Solver<Dtype>(param);                                 \
    }                                                                          \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)                          

    
} //namespace caffe




#endif
