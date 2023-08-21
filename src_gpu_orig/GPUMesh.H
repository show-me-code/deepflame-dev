#include <iostream>

class GPUMesh
{
public:
    int num_cells;
    int num_faces;
    int boundary_cells;
    //... all variables needed to upload once

public:
    GPUMesh();
    ~GPUMesh();
};

GPUMesh::GPUMesh()
{
    // same to the constructor of dfMatrix.C
}

GPUMesh::~GPUMesh()
{
}
