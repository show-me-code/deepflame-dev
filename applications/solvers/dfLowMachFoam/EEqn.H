{
    volScalarField& he = thermo.he();

#ifdef GPUSolver_
        start1 = std::clock();
        UEqn_GPU.updatePsi(&U[0][0]);
        K = 0.5*magSqr(U);
        end1 = std::clock();
        time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
        time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
        time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

#ifdef CPUSolver_
    start1 = std::clock();
    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + fvm::div(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      - fvm::laplacian(turbulence->alphaEff(), he)
      + diffAlphaD
     ==
        fvc::div(hDiffCorrFlux)
    );

    EEqn.relax();

    EEqn.solve();

    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

#ifdef GPUSolver_
    // prepare data on CPU
    start1 = std::clock();
    start2 = std::clock();
    const tmp<volScalarField> alphaEff_tmp(turbulence->alphaEff());
    const volScalarField& alphaEff = alphaEff_tmp();
    int eeqn_offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        int patchSize = pw.size();

        const scalarField& patchK = K.boundaryField()[patchi];
        const vectorField& patchhDiffCorrFlux = hDiffCorrFlux.boundaryField()[patchi];
        const scalarField& patchAlphaEff = alphaEff.boundaryField()[patchi];
        memcpy(boundary_K + eeqn_offset, &patchK[0], patchSize*sizeof(double));
        memcpy(boundary_hDiffCorrFlux + eeqn_offset * 3, &patchhDiffCorrFlux[0][0], 3 * patchSize*sizeof(double));
        memcpy(boundary_alphaEff + eeqn_offset, &patchAlphaEff[0], patchSize*sizeof(double));

        Field<scalar> valueInternalCoeffs = patchFlux*he.boundaryField()[patchi].valueInternalCoeffs(pw);
        Field<scalar> valueBoundaryCoeffs = -patchFlux*he.boundaryField()[patchi].valueBoundaryCoeffs(pw);
        Field<scalar> gradientInternalCoeffs = he.boundaryField()[patchi].gradientInternalCoeffs();
        Field<scalar> gradientBoundaryCoeffs = he.boundaryField()[patchi].gradientBoundaryCoeffs();
        memcpy(eeqn_valueInternalCoeffs + eeqn_offset, &valueInternalCoeffs[0], patchSize*sizeof(double));
        memcpy(eeqn_valueBoundaryCoeffs + eeqn_offset, &valueBoundaryCoeffs[0], patchSize*sizeof(double));
        memcpy(eeqn_gradientInternalCoeffs + eeqn_offset, &gradientInternalCoeffs[0], patchSize*sizeof(double));
        memcpy(eeqn_gradientBoundaryCoeffs + eeqn_offset, &gradientBoundaryCoeffs[0], patchSize*sizeof(double));

        eeqn_offset += patchSize;
    }
    eeqn_offset = 0;
    end1 = std::clock();
    time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly_CPU_Prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // prepare data on GPU
    start1 = std::clock();
    EEqn_GPU.prepare_data(&he.oldTime()[0], &K[0], &K.oldTime()[0], &alphaEff[0],
            &dpdt[0], &diffAlphaD[0], &hDiffCorrFlux[0][0],
            boundary_K, boundary_hDiffCorrFlux, boundary_alphaEff,
            eeqn_valueInternalCoeffs, eeqn_valueBoundaryCoeffs,
            eeqn_gradientInternalCoeffs, eeqn_gradientBoundaryCoeffs);
    if (doSync) EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn_mtxAssembly_GPU_Prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    EEqn_GPU.resetAb();
    EEqn_GPU.fvm_ddt();
    EEqn_GPU.fvm_div();
    EEqn_GPU.fvm_laplacian();
    EEqn_GPU.fvc_ddt();
    EEqn_GPU.fvc_div_phi_scalar();
    EEqn_GPU.fvc_div_vector();
    EEqn_GPU.add_to_source();
    if (doSync) EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn_mtxAssembly_GPU_Run += double(end1 - start1) / double(CLOCKS_PER_SEC);

    EEqn_GPU.sync();
    end2 = std::clock();
    time_monitor_EEqn += double(end2 - start2) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end2 - start2) / double(CLOCKS_PER_SEC);

    // check value of mtxAssembly, no time monitor
    EEqn_GPU.checkValue(false);

    start1 = std::clock();
    EEqn_GPU.solve();
    if (doSync) EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    EEqn_GPU.updatePsi(&he[0]);
    end1 = std::clock();
    time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif
}
