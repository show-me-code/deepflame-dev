// Solve the Momentum equation
#ifdef GPUSolver_
    start1 = std::clock();
    UEqn_GPU.fvm_ddt(&U.oldTime()[0][0]);
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
        memcpy(ueqn_internalCoeffs_init + 3*offset, &ueqn_internalCoeffs_vec[0][0], 3*patchSize*sizeof(double));
        // need to construct every time step
        memcpy(ueqn_boundaryCoeffs_init + 3*offset, &ueqn_boundaryCoeffs_vec[0][0], 3*patchSize*sizeof(double));
        // laplacian internalCoeffs
        memcpy(ueqn_laplac_internalCoeffs_init + 3*offset, &ueqn_laplac_internalCoeffs_vec[0][0], 3*patchSize*sizeof(double));
        // laplacian boundaryCoeffs
        memcpy(ueqn_laplac_boundaryCoeffs_init + 3*offset, &ueqn_laplac_boundaryCoeffs_vec[0][0], 3*patchSize*sizeof(double));
        // boundary pressure
        memcpy(boundary_pressure_init+offset, &patchP[0], patchSize*sizeof(double));
        // boundary velocity
        memcpy(boundary_velocity_init+3*offset, &patchU[0][0], 3*patchSize*sizeof(double));
        // boundary nuEff
        memcpy(boundary_nuEff_init+offset, &patchNuEff[0], patchSize*sizeof(double));
        // boundary rho
        memcpy(boundary_rho_init+offset, &patchRho[0], patchSize*sizeof(double));
        offset += patchSize;
    }
    end1 = std::clock();
    end2 = std::clock();
    time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    UEqn_GPU.fvm_div(&phi[0], ueqn_internalCoeffs_init, ueqn_boundaryCoeffs_init, 
        ueqn_laplac_internalCoeffs_init, ueqn_laplac_boundaryCoeffs_init, 
        boundary_pressure_init, boundary_velocity_init, boundary_nuEff_init, boundary_rho_init);
    UEqn_GPU.fvc_grad(&p[0]);
    UEqn_GPU.fvc_grad_vector();
    UEqn_GPU.dev2T();
    UEqn_GPU.fvc_div_tensor(&nuEff[0]);
    UEqn_GPU.fvm_laplacian();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // start2 = std::clock();
    // fvVectorMatrix turb_source
    // (
    //     turbulence->divDevRhoReff(U)
    // );
    // end2 = std::clock();
    // time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);

    // UEqn_GPU.add_fvMatrix(&turb_source.lower()[0], &turb_source.diag()[0], &turb_source.upper()[0], &turb_source.source()[0][0]);
    // end1 = std::clock();
    // time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    // time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // check value
    // if (pimple.momentumPredictor())
    // {
    //     solve(UEqn);

    //     K = 0.5*magSqr(U);
    // }
    // UEqn_GPU.checkValue(false);
#else
    start1 = std::clock();
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::ddt(rho, U) + fvm::div(phi, U)
    + turbulence->divDevRhoReff(U) 
    == -fvc::grad(p)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // UEqn.relax();
    start1 = std::clock();
    if (pimple.momentumPredictor())
    {
        solve(UEqn);

        K = 0.5*magSqr(U);
    }
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

// start1 = std::clock();
// // // std::thread t(&dfMatrix::solve, &UEqn_GPU);
// UEqn_GPU.solve();
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

// start1 = std::clock();
// // // t.join();
// // UEqn_GPU.updatePsi(&U[0][0]);
// K = 0.5*magSqr(U);
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
// // Info << "U_amgx = " << U << endl;

