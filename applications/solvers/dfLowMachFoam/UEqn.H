// Solve the Momentum equation
// start1 = std::clock();
// Info << "U_of = " << U << endl;
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
  + turbulence->divDevRhoReff(U) 
  == -fvc::grad(p)
);
fvVectorMatrix& UEqn = tUEqn.ref();

const tmp<volTensorField> tgradU(fvc::grad(U));
const volTensorField& gradU = tgradU();

// const tmp<volTensorField> ttest(turbulence->nuEff()*turbulence->alpha()*rho()*dev2(T(fvc::grad(U))));
// const tmp<volTensorField> ttest(rho()*dev2(Foam::T(fvc::grad(U))));
// const volTensorField& testU = ttest();

// print boundaryfield of gradU
std::vector<double> tmp_boundary_gradU;
int patchSize = 0;
forAll(U.boundaryField(), patchi)
{
    const tensorField& pSf = gradU.boundaryField()[patchi];
    patchSize = pSf.size();
    tmp_boundary_gradU.insert(tmp_boundary_gradU.end(), &pSf[0][0], &pSf[0][0]+9*patchSize);
}

// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

// start1 = std::clock();
// fvVectorMatrix turb_source
// (
//     turbulence->divDevRhoReff(U)
// );
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);


// UEqn.relax();
// start1 = std::clock();
// if (pimple.momentumPredictor())
// {
//     solve(UEqn);

//     K = 0.5*magSqr(U);
// }
// Info << "U_of = " << U << endl;
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);


start1 = std::clock();

UEqn_GPU.fvm_ddt(&rho.oldTime()[0], &rho[0], &U.oldTime()[0][0]);

start2 = std::clock();
int offset = 0;
const tmp<volScalarField> nuEff_tmp(turbulence->nuEff());
const volScalarField& nuEff = nuEff_tmp();
forAll(U.boundaryField(), patchi)
{
    const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
    const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];

    const scalarField& patchP = p.boundaryField()[patchi];
    const vectorField& pSf = mesh.Sf().boundaryField()[patchi];
    const vectorField& patchU = U.boundaryField()[patchi];
    const scalarField& patchRho = rho.boundaryField()[patchi];
    const scalarField& patchNuEff = nuEff.boundaryField()[patchi];

    int patchSize = pw.size();

    Field<vector> ueqn_internalCoeffs_vec = patchFlux*U.boundaryField()[patchi].valueInternalCoeffs(pw);
    Field<vector> ueqn_boundaryCoeffs_vec = -patchFlux*U.boundaryField()[patchi].valueBoundaryCoeffs(pw); 
    Field<vector> ueqn_laplac_internalCoeffs_vec = U.boundaryField()[patchi].gradientInternalCoeffs();
    Field<vector> ueqn_laplac_boundaryCoeffs_vec = U.boundaryField()[patchi].gradientBoundaryCoeffs();

    // only need to construct once
    std::copy(&ueqn_internalCoeffs_vec[0][0], &ueqn_internalCoeffs_vec[0][0]+3*patchSize, ueqn_internalCoeffs_init + 3*offset);

    // need to construct every time step
    std::copy(&ueqn_boundaryCoeffs_vec[0][0], &ueqn_boundaryCoeffs_vec[0][0]+3*patchSize, ueqn_boundaryCoeffs_init + 3*offset);

    // laplacian internalCoeffs
    std::copy(&ueqn_laplac_internalCoeffs_vec[0][0], &ueqn_laplac_internalCoeffs_vec[0][0]+3*patchSize, ueqn_laplac_internalCoeffs_init + 3*offset);
    
    // laplacian boundaryCoeffs
    std::copy(&ueqn_laplac_boundaryCoeffs_vec[0][0], &ueqn_laplac_boundaryCoeffs_vec[0][0]+3*patchSize, ueqn_laplac_boundaryCoeffs_init + 3*offset);

    // boundary pressure
    std::copy(&patchP[0], &patchP[0]+patchSize, boundary_pressure_init+offset);

    // boundary velocity
    std::copy(&patchU[0][0], &patchU[0][0]+3*patchSize, boundary_velocity_init+3*offset);

    // boundary nuEff
    std::copy(&patchNuEff[0], &patchNuEff[0]+patchSize, boundary_nuEff_init+offset);

    // boundary rho
    std::copy(&patchRho[0], &patchRho[0]+patchSize, boundary_rho_init+offset);

    offset += patchSize;
}

// test boundary of of output 
FILE *fp = NULL;
char *input_file = "of_ref_boundary.txt";
fp = fopen(input_file, "rb+");
double *of_ref_boundary = new double[9*num_boundary_faces];
int readfile = 0;
offset = 0;

forAll(U.boundaryField(), patchi)
{
    int patchSize = U.boundaryField()[patchi].size();
    readfile = fread(of_ref_boundary+9*offset, sizeof(double), patchSize*9, fp);
    offset += patchSize;
}
char *input_file1 = "of_ref_fvc_grad.txt";
fp = fopen(input_file1, "rb+");
double *of_ref_grad = new double[9*num_cells];
readfile = fread(of_ref_grad, sizeof(double), 9*num_cells, fp);

char *input_file2 = "of_ref_fvc_grad_T.txt";
fp = fopen(input_file2, "rb+");
double *of_ref_grad_T = new double[9*num_cells];
readfile = fread(of_ref_grad_T, sizeof(double), 9*num_cells, fp);

char *input_file3 = "of_ref_fvc_div.txt";
fp = fopen(input_file3, "rb+");
double *of_ref_div = new double[3*num_cells];
readfile = fread(of_ref_div, sizeof(double), 3*num_cells, fp);

end2 = std::clock();
time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);

UEqn_GPU.fvm_div(&phi[0], ueqn_internalCoeffs_init, ueqn_boundaryCoeffs_init, 
ueqn_laplac_internalCoeffs_init, ueqn_laplac_boundaryCoeffs_init, 
boundary_pressure_init, boundary_velocity_init, boundary_nuEff_init, boundary_rho_init);

UEqn_GPU.fvc_grad(&p[0]);

UEqn_GPU.fvc_grad_vector(tmp_boundary_gradU, &gradU[0][0]);
UEqn_GPU.divDevRhoReff(of_ref_grad_T);
UEqn_GPU.fvc_div_tensor(&nuEff[0], of_ref_boundary, of_ref_grad, of_ref_div);
UEqn_GPU.fvm_laplacian();

start2 = std::clock();
fvVectorMatrix turb_source
(
    turbulence->divDevRhoReff(U)
);
end2 = std::clock();
time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);

// UEqn_GPU.add_fvMatrix(&turb_source.lower()[0], &turb_source.diag()[0], &turb_source.upper()[0], &turb_source.source()[0][0]);
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

// UEqn_GPU.checkValue(false);

start1 = std::clock();
// std::thread t(&dfMatrix::solve, &UEqn_GPU);
UEqn_GPU.solve();
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

start1 = std::clock();
// t.join();
UEqn_GPU.updatePsi(&U[0][0]);
K = 0.5*magSqr(U);
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
// Info << "U_amgx = " << U << endl;

