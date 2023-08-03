#ifdef GPUSolver_
// run CPU
tmp<fvVectorMatrix> tUEqn
(
 fvm::div(phi, U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

// run GPU
// preProcess
// skip preparing u, p, nu_eff, and rho.boundary used by UEqn_GPU.preProcess()
// TODO: temp, now we only transfer phi(instead of rhoEqn) used by fvm::div(phi, U)
double *h_u = dfDataBase.getFieldPointer("u", location::cpu, position::internal);
double *h_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::internal);
double *h_boundary_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::boundary);
memcpy(h_phi, &phi[0], dfDataBase.surface_value_bytes);
int offset = 0;
forAll(phi.boundaryField(), patchi)
{
    const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
    int patchsize = patchPhi.size();
    memcpy(h_boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
    offset += patchsize;
}
UEqn_GPU.preProcessForRhoEqn(h_phi, h_boundary_phi);
// process
UEqn_GPU.process();
// postProcess
UEqn_GPU.postProcess(h_u);
// checkResult
// TODO: temp, now we compare ldu, finally we compare csr
std::vector<double> h_internal_coeffs(dfDataBase.num_boundary_surfaces * 3);
std::vector<double> h_boundary_coeffs(dfDataBase.num_boundary_surfaces * 3);
offset = 0;
for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
{
    int patchsize = dfDataBase.patch_size[patchi];
    const double* internal_coeff_ptr = &UEqn.internalCoeffs()[patchi][0][0];
    const double* boundary_coeff_ptr = &UEqn.boundaryCoeffs()[patchi][0][0];
    memcpy(h_internal_coeffs.data() + offset * 3, internal_coeff_ptr, patchsize * 3 * sizeof(double));
    memcpy(h_boundary_coeffs.data() + offset * 3, boundary_coeff_ptr, patchsize * 3 * sizeof(double));
    offset += patchsize;
}
bool printFlag = false;
UEqn_GPU.compareResult(&UEqn.lower()[0], &UEqn.upper()[0], &UEqn.diag()[0],
        h_internal_coeffs.data(), h_boundary_coeffs.data(), printFlag);
#endif
