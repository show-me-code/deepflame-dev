#include "dfRhoEqn.H"

void dfRhoEqn::createNonConstantLduAndCsrFields()
{
}

void dfRhoEqn::preProcess(const double *h_phi, const double *h_boundary_phi)
{
}

void dfRhoEqn::process()
{
    //使用event计算时间
    float time_elapsed=0;
    cudaEvent_t start,stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start,0));

#ifndef TIME_GPU
    if(!graph_created) {
        DEBUG_TRACE;
        checkCudaErrors(cudaStreamBeginCapture(dataBase_.stream, cudaStreamCaptureModeGlobal));
#endif

    checkCudaErrors(cudaMallocAsync((void**)&d_source, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMallocAsync((void**)&d_diag, dataBase_.cell_value_bytes, dataBase_.stream));

    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_phi, dataBase_.h_phi, dataBase_.surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));
    checkCudaErrors(cudaMemcpyAsync(dataBase_.d_boundary_phi, dataBase_.h_boundary_phi, dataBase_.boundary_surface_value_bytes, cudaMemcpyHostToDevice, dataBase_.stream));

    checkCudaErrors(cudaMemsetAsync(d_diag, 0, dataBase_.cell_value_bytes, dataBase_.stream));
    checkCudaErrors(cudaMemsetAsync(d_source, 0, dataBase_.cell_value_bytes, dataBase_.stream));

    fvm_ddt_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.rdelta_t, dataBase_.d_rho_old, dataBase_.d_volume, 
            d_diag, d_source);
    fvc_div_surface_scalar(dataBase_.stream, dataBase_.num_cells, dataBase_.num_surfaces, dataBase_.num_boundary_surfaces, dataBase_.d_owner,
            dataBase_.d_neighbor, dataBase_.d_phi, dataBase_.d_boundary_face_cell, dataBase_.d_boundary_phi, dataBase_.d_volume, d_source, -1);
    solve();
#ifndef TIME_GPU
        checkCudaErrors(cudaStreamEndCapture(dataBase_.stream, &graph));
        checkCudaErrors(cudaGraphInstantiate(&graph_instance, graph, NULL, NULL, 0));
        graph_created = true;
    }
    checkCudaErrors(cudaGraphLaunch(graph_instance, dataBase_.stream));
#endif

    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(start));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_elapsed,start,stop));
    fprintf(stderr, "rhoEqn process time:%f(ms)\n",time_elapsed);
}

void dfRhoEqn::sync()
{
    checkCudaErrors(cudaStreamSynchronize(dataBase_.stream));
}

void dfRhoEqn::solve()
{
    solve_explicit_scalar(dataBase_.stream, dataBase_.num_cells, d_diag, d_source, dataBase_.d_rho);
}

void dfRhoEqn::postProcess(double *h_rho)
{
    checkCudaErrors(cudaFreeAsync(d_source, dataBase_.stream));
    checkCudaErrors(cudaFreeAsync(d_diag, dataBase_.stream));

    checkCudaErrors(cudaMemcpyAsync(h_rho, dataBase_.d_rho, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost, dataBase_.stream));
    sync();
    // TODO: correct boundary conditions
}

void dfRhoEqn::compareResult(const double *diag, const double *source, bool printFlag)
{
    std::vector<double> h_diag;
    h_diag.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_diag.data(), d_diag, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_diag");
    checkVectorEqual(dataBase_.num_cells, diag, h_diag.data(), 1e-14, printFlag);
    DEBUG_TRACE;

    std::vector<double> h_source;
    h_source.resize(dataBase_.num_cells);
    checkCudaErrors(cudaMemcpy(h_source.data(), d_source, dataBase_.cell_value_bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "check h_source");
    checkVectorEqual(dataBase_.num_cells, source, h_source.data(), 1e-14, printFlag);
    DEBUG_TRACE;
}
