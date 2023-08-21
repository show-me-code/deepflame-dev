#ifdef GPUSolver_
const tmp<volScalarField> nuEff_tmp(turbulence->nuEff());
const volScalarField& nuEff = nuEff_tmp();

// run CPU, for temp
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
    +  turbulence->divDevRhoReff(U)
    == -fvc::grad(p)
);
// tmp<fvVectorMatrix> tUEqn_ref // test turbulence->divDevRhoReff(U)
// (
//     - fvc::div((turbulence->rho()*turbulence->nuEff())*dev2(Foam::T(fvc::grad(U))))
//     - fvm::laplacian(turbulence->rho()*turbulence->nuEff(), U)
// );

fvVectorMatrix& UEqn = tUEqn.ref();

// run GPU
// preProcess
// TODO: preProcessForRhoEqn for temp, now we only transfer phi(instead of rhoEqn) used by fvm::div(phi, U)
UEqn_GPU.sync();
double *h_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::internal);
double *h_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::internal);
double *h_boundary_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::boundary);
memcpy(h_rho, &rho[0], dfDataBase.cell_value_bytes);
memcpy(h_phi, &phi[0], dfDataBase.surface_value_bytes);
int offset = 0;
forAll(phi.boundaryField(), patchi)
{
    const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
    int patchsize = patchPhi.size();
    memcpy(h_boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
    offset += patchsize;
}
UEqn_GPU.preProcessForRhoEqn(h_rho, h_phi, h_boundary_phi);
DEBUG_TRACE;
clock_t start = std::clock();
// preparing u, p, nu_eff, and rho.boundary used by UEqn_GPU.preProcess()
double *h_u = dfDataBase.getFieldPointer("u", location::cpu, position::internal);
double *h_boundary_u = dfDataBase.getFieldPointer("u", location::cpu, position::boundary);
double *h_p = dfDataBase.getFieldPointer("p", location::cpu, position::internal);
double *h_boundary_p = dfDataBase.getFieldPointer("p", location::cpu, position::boundary);
double *h_nu_eff = UEqn_GPU.getFieldPointer("nu_eff", location::cpu, position::internal);
double *h_boundary_nu_eff = UEqn_GPU.getFieldPointer("nu_eff", location::cpu, position::boundary);
double *h_boundary_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::boundary);
double end = std::clock();
Info << "get pointer" << double(end - start) / double(CLOCKS_PER_SEC) << endl;

start = std::clock();
memcpy(h_u, &U.oldTime()[0][0], dfDataBase.cell_value_vec_bytes);
memcpy(h_p, &p[0], dfDataBase.cell_value_bytes);
memcpy(h_nu_eff, &nuEff[0], dfDataBase.cell_value_bytes);
end = std::clock();
Info << "copy to pinned memory" << double(end - start) / double(CLOCKS_PER_SEC) << endl;

start = std::clock();
offset = 0;
forAll(U.boundaryField(), patchi)
{
    const fvPatchVectorField& patchU = U.boundaryField()[patchi];
    const fvPatchScalarField& patchP = p.boundaryField()[patchi];
    const fvPatchScalarField& patchNuEff = nuEff.boundaryField()[patchi];
    const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
    int patchsize = patchU.size();
    memcpy(h_boundary_u + offset * 3, &patchU[0][0], patchsize * 3 * sizeof(double));
    memcpy(h_boundary_p + offset, &patchP[0], patchsize * sizeof(double));
    memcpy(h_boundary_nu_eff + offset, &patchNuEff[0], patchsize * sizeof(double));
    memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
    offset += patchsize;
}
end = std::clock();
Info << "CPU prepare boundary time" << double(end - start) / double(CLOCKS_PER_SEC) << endl;

start = std::clock();
UEqn_GPU.preProcess(h_u, h_boundary_u, h_p, h_boundary_p, h_nu_eff, h_boundary_nu_eff, h_boundary_rho);
DEBUG_TRACE;
UEqn_GPU.sync();
end = std::clock();
Info << "GPU preProcess time" << double(end - start) / double(CLOCKS_PER_SEC) << endl;

// process
start = std::clock();
UEqn_GPU.process();
end = std::clock();
DEBUG_TRACE;
UEqn_GPU.sync();
// end = std::clock();
Info << "GPU process time" << double(end - start) / double(CLOCKS_PER_SEC) << endl;

// postProcess
UEqn_GPU.postProcess(h_u);
DEBUG_TRACE;

// checkResult
// TODO: for temp, now we compare ldu, finally we compare csr
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
bool printFlag = true;
UEqn_GPU.compareResult(&UEqn.lower()[0], &UEqn.upper()[0], &UEqn.diag()[0], &UEqn.source()[0][0],
        h_internal_coeffs.data(), h_boundary_coeffs.data(), 
        // &DivTensor[0][0], 
        printFlag);
DEBUG_TRACE;
#endif
