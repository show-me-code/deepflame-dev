#pragma once
#include <cstdint>
#include <cstdbool>
#include <numeric>
#include <string>
#include <vector>

template<typename DataType>
class Tensor{
private:
    std::vector<int64_t> shape_;
    int64_t element_num_;
    DataType* data_;
    bool owner_;
public:
    Tensor(std::vector<int64_t> shape):
        shape_(shape),
        element_num_(std::accumulate(shape_.begin(),shape_.end(),1,std::multiplies<int64_t>())),
        data_(new DataType[element_num_]),
        owner_(true){};

    Tensor(std::vector<int64_t> shape, DataType* data):
        shape_(shape), 
        element_num_(std::accumulate(shape_.begin(),shape_.end(),1,std::multiplies<int64_t>())),
        data_(data), 
        owner_(false){};
    ~Tensor(){
        if(owner_)  delete[] data_;
    }
    int64_t dim_num() const {return shape_.size();};
    int64_t element_num() const {return element_num_;};
    int64_t bytes_num() const {return element_num_ * sizeof(DataType);};
    int64_t dim(int64_t i) const {return shape_[i];};
    const DataType* data() const {return data_;};
    DataType* data() {return data_;};
};

template<typename DataType>
class Layer{
public:
    virtual ~Layer() {};
    virtual void forward(const Tensor<DataType>& input, Tensor<DataType>& output) = 0;
    virtual void load_parameters(const std::string& dir, int64_t layer_id) = 0;
};

template<typename DataType>
class Linear : public Layer<DataType>{
private:
    int64_t in_features_;
    int64_t out_features_;
    // (in_features, out_features)
    Tensor<DataType> weights_;
    // (out_features,)
    Tensor<DataType> bias_;

public:
    Linear(int64_t in_features,int64_t out_features):
        in_features_(in_features),
        out_features_(out_features),
        weights_({in_features_,out_features_}),
        bias_({out_features_}){};
    virtual ~Linear(){};

    virtual void forward(const Tensor<DataType>& input, Tensor<DataType>& output);
    virtual void load_parameters(const std::string& dir, int64_t layer_id);
};

template<typename DataType>
class LinearGELU : public Layer<DataType>{
private:
    int64_t in_features_;
    int64_t out_features_;
    // (in_features, out_features)
    Tensor<DataType> weights_;
    // (out_features,)
    Tensor<DataType> bias_;

public:
    LinearGELU(int64_t in_features,int64_t out_features):
        in_features_(in_features),
        out_features_(out_features),
        weights_({in_features_,out_features_}),
        bias_({out_features_}){};
    virtual ~LinearGELU(){};

    virtual void forward(const Tensor<DataType>& input, Tensor<DataType>& output);
    virtual void load_parameters(const std::string& dir, int64_t layer_id);
};