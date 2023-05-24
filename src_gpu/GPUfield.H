/*
GPU field
UEqn
GPUField
rho_old, rho_new, vector_old, phi, p, nuEff, 
p_bou, vector_old_bou, nuEff_bou, rho_new_bou, 
ueqn_internalCoeffs, ueqn_boundaryCoeffs_init, ueqn_laplac_internalCoeffs_init, ueqn_laplac_boundaryCoeffs_init
1. initialize a map
2. implement methods to construct GPU pointer and return GPU pointer

fvm__ddt(UEqn, rho, U)
{
    rho.oldtime()
    rho.
}
*/
#include <iostream>
#include <unordered_map>

#define TOSTRING(x) #x

struct GPUField {
    double* cur_internal;
    double* cur_boundary;
    double* old_internal;
    double* old_boundary;
};
std::unordered_map<std::string, GPUField> GPUFields;

// initialize: cudaMalloc, conducted at begining
void initialize(std::string U, std::string p, std::string phi);
// Q: 
// 1. consider the different sizes bettween face values and cell values
// 2. not all variables need all these four terms

// update at the end of this time step
// move current pointer as oldTime pointer
void update();

template<class Type, template<class> class PatchField, class GeoMesh>
double* cur_internal(Foam::GeometryField<Type, PatchField, GeoMesh> var)
{
    if (!GPUFields[TOSTRING(var)].cur_internal) {
        // 1. cudaMemcopy current internal field
    }
    return GPUFields[TOSTRING(var)].cur_internal;
}

template<class Type, template<class> class PatchField, class GeoMesh>
double* cur_boundary(Foam::GeometryField<Type, PatchField, GeoMesh> var);

template<class Type, template<class> class PatchField, class GeoMesh>
double* old_internal(Foam::GeometryField<Type, PatchField, GeoMesh> var);

template<class Type, template<class> class PatchField, class GeoMesh>
double* old_boundary(Foam::GeometryField<Type, PatchField, GeoMesh> var);

