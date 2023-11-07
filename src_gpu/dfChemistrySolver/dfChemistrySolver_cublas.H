#pragma once

#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <torch/script.h>

class dfChemistrySolver
{
private:
    std::vector<torch::jit::script::Module> modules_;
    torch::Device device_;
    double *Xmu_, *Xstd_, *Ymu_, *Ystd_;

    double *init_input_, *y_input_BCT;
    int dim_input_, num_cells_, num_species_, num_modules_;
public:
    dfChemistrySolver(int num_cells, int num_species);
    ~dfChemistrySolver();

    void loadModels(const std::string dir);
    void loadNormalization(const std::string dir);
    void Inference(const double *T, const double *p, const double *y,
            const double *rho, double *RR);
};