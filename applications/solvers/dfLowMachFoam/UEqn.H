// Solve the Momentum equation

tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(rho, U) + fvm::div(phi, U)
  + turbulence->divDevRhoReff(U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    K = 0.5*magSqr(U);
}


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
    // boundary face vector
    std::copy(&pSf[0][0], &pSf[0][0]+3*patchSize, boundary_face_vector_init.begin()+3*offset);

    offset += patchSize;
}
fvVectorMatrix turb_source
(
    turbulence->divDevRhoReff(U)
);

UEqn_GPU.fvm_ddt(&rho.oldTime()[0], &rho[0], &mesh.V()[0], &U.oldTime()[0][0]);
UEqn_GPU.fvm_div(&mesh.surfaceInterpolation::weights()[0], &phi[0], ueqn_internalCoeffs_init, ueqn_boundaryCoeffs_init);
UEqn_GPU.fvc_grad(&mesh.Sf()[0][0], &p[0], boundary_face_vector_init, boundary_pressure_init);
UEqn_GPU.add_fvMatrix(&turb_source.lower()[0], &turb_source.diag()[0], &turb_source.upper()[0], &turb_source.source()[0][0]);
UEqn_GPU.print();