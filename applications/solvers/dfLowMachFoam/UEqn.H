// Solve the Momentum equation
// start1 = std::clock();
// Info << "U_of = " << U << endl;
tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
  + turbulence->divDevRhoReff(U) == -fvc::grad(p)
);
fvVectorMatrix& UEqn = tUEqn.ref();
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
forAll(U.boundaryField(), patchi)
{
    const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
    const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];

    const scalarField& patchP = p.boundaryField()[patchi];
    const vectorField& pSf = mesh.Sf().boundaryField()[patchi];

    int patchSize = pw.size();

    Field<vector> ueqn_internalCoeffs_vec = patchFlux*U.boundaryField()[patchi].valueInternalCoeffs(pw);
    Field<vector> ueqn_boundaryCoeffs_vec = -patchFlux*U.boundaryField()[patchi].valueBoundaryCoeffs(pw); 

    // only need to construct once
    std::copy(&ueqn_internalCoeffs_vec[0][0], &ueqn_internalCoeffs_vec[0][0]+3*patchSize, ueqn_internalCoeffs_init.begin() + 3*offset);

    // need to construct every time step
    std::copy(&ueqn_boundaryCoeffs_vec[0][0], &ueqn_boundaryCoeffs_vec[0][0]+3*patchSize, ueqn_boundaryCoeffs_init.begin() + 3*offset);

    // boundary pressure
    std::copy(&patchP[0], &patchP[0]+patchSize, boundary_pressure_init.begin()+offset);

    offset += patchSize;
}
end2 = std::clock();
time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);

UEqn_GPU.fvm_div(&phi[0], ueqn_internalCoeffs_init, ueqn_boundaryCoeffs_init, boundary_pressure_init);
UEqn_GPU.fvc_grad(&p[0]);

start2 = std::clock();
fvVectorMatrix turb_source
(
    turbulence->divDevRhoReff(U)
);
end2 = std::clock();
time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);

UEqn_GPU.add_fvMatrix(&turb_source.lower()[0], &turb_source.diag()[0], &turb_source.upper()[0], &turb_source.source()[0][0]);
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

// UEqn_GPU.checkValue(false);

start1 = std::clock();
UEqn_GPU.solve();
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

start1 = std::clock();
UEqn_GPU.updatePsi(&U[0][0]);
K = 0.5*magSqr(U);
end1 = std::clock();
time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
// Info << "U_amgx = " << U << endl;

