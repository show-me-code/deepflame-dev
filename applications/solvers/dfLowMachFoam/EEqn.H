{
    fprintf(stderr, "\n\n=====================enter EEqn=====================\n\n\n");
    volScalarField& he = thermo.he();
    #ifdef GPUSolver_
        // start1 = std::clock();
        // t.join();
        // UEqn_GPU.updatePsi(&U[0][0]);
        // K = 0.5*magSqr(U);
        // end1 = std::clock();
        // time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
        // time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
        // time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
        // time_monitor_UinE += double(end1 - start1) / double(CLOCKS_PER_SEC);
        start1 = std::clock();
        // // t.join();
        UEqn_GPU.updatePsi(&U[0][0]);
        K = 0.5*magSqr(U);
        // K = 0.5*magSqr(U) * 10000000000;
        end1 = std::clock();
        time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
        time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
        time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
    #endif
#ifdef CPUSolver_
    //fvScalarMatrix EEqn
    //(
    //    fvm::ddt(rho, he) + fvm::div(phi, he)
    //  + fvc::ddt(rho, K) + fvc::div(phi, K)
    //  - dpdt
    //  - fvm::laplacian(turbulence->alphaEff(), he)
    //  + diffAlphaD
    // ==
    //    fvc::div(hDiffCorrFlux)
    //);

    //EEqn.relax();

    //EEqn.solve();

    fvScalarMatrix EEqn
    (
        //fvm::ddt(rho, he) + fvm::div(phi, he)
        //fvm::ddt(rho, he) + fvc::ddt(rho, K)
        //fvm::ddt(rho, he) + fvc::div(phi, K) * 10000000000
        //fvm::ddt(rho, he) + fvc::div(phi, K)
        //fvm::laplacian(turbulence->alphaEff(), he)
        fvm::div(phi, he)
        //fvm::laplacian(turbulence->alphaEff(), he) - dpdt + diffAlphaD
        //fvm::laplacian(turbulence->alphaEff(), he) - fvc::div(hDiffCorrFlux)
        //fvm::laplacian(turbulence->alphaEff(), he) + fvc::div(phi, K) * 10000000000
        //==
        //fvc::div(hDiffCorrFlux)
    );
    EEqn.relax();
    EEqn.solve();
#endif

#ifdef GPUSolver_
    start2 = std::clock();
    // prepare data on CPU
    const tmp<volScalarField> alphaEff_tmp(turbulence->alphaEff());
    const volScalarField& alphaEff = alphaEff_tmp();

    int idx = 0;
    offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        int patchSize = pw.size();

        const scalarField& patchK = K.boundaryField()[patchi];
        const scalarField& patchAlphaEff = alphaEff.boundaryField()[patchi];
        memcpy(boundary_K + offset, &patchK[0], patchSize*sizeof(double));
        memcpy(boundary_alphaEff + offset, &patchAlphaEff[0], patchSize*sizeof(double));

        Field<scalar> valueInternalCoeffs = patchFlux*he.boundaryField()[patchi].valueInternalCoeffs(pw);
        Field<scalar> valueBoundaryCoeffs = -patchFlux*he.boundaryField()[patchi].valueBoundaryCoeffs(pw);
        Field<scalar> gradientInternalCoeffs = he.boundaryField()[patchi].gradientInternalCoeffs();
        Field<scalar> gradientBoundaryCoeffs = he.boundaryField()[patchi].gradientBoundaryCoeffs();
        memcpy(eeqn_valueInternalCoeffs + offset, &valueInternalCoeffs[0], patchSize*sizeof(double));
        memcpy(eeqn_valueBoundaryCoeffs + offset, &valueBoundaryCoeffs[0], patchSize*sizeof(double));
        memcpy(eeqn_gradientInternalCoeffs + offset, &gradientInternalCoeffs[0], patchSize*sizeof(double));
        memcpy(eeqn_gradientBoundaryCoeffs + offset, &gradientBoundaryCoeffs[0], patchSize*sizeof(double));

        //for (int t = 0; t < patchSize; t++) {
        //    fprintf(stderr, "gpu valueInternalCoeffs[%d][%d]: %.10lf\n", patchi, t, eeqn_valueInternalCoeffs[offset + t]);
        //    if (idx == 64 || idx == 4160 || idx == 8385)
        //        fprintf(stderr, "eeqn_value_internal_coeffs[%d](other index: %d %d): %.10lf(other value: %.10lf)\n", idx, offset, t, eeqn_valueInternalCoeffs[idx], eeqn_valueInternalCoeffs[offset + t]);
        //    idx++;
        //}
        offset += patchSize;
    }
    offset = 0;
    end2 = std::clock();
    time_monitor_CPU += double(end2 - start2) / double(CLOCKS_PER_SEC);
    start1 = std::clock();
    // prepare data on GPU
    EEqn_GPU.prepare_data(&he.oldTime()[0], &K[0], &K.oldTime()[0], &alphaEff[0],
            &dpdt[0], &diffAlphaD[0], &hDiffCorrFlux[0][0],
            boundary_K, boundary_alphaEff,
            eeqn_valueInternalCoeffs, eeqn_valueBoundaryCoeffs,
            eeqn_gradientInternalCoeffs, eeqn_gradientBoundaryCoeffs);
    end1 = std::clock();
    time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    fprintf(stderr, "\n\n=====================EEqn GPUSolver_ part2=====================\n\n\n");

    start1 = std::clock();
    EEqn_GPU.resetAb();
    //EEqn_GPU.fvm_ddt();
    EEqn_GPU.fvm_div();
    //EEqn_GPU.fvm_laplacian();
    //EEqn_GPU.fvc_ddt();
    //EEqn_GPU.fvc_div_phi_scalar();
    //EEqn_GPU.fvc_div_vector();
    //EEqn_GPU.add_to_source();
    EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    fprintf(stderr, "\n\n=====================EEqn GPUSolver_ part3=====================\n\n\n");

    // check value of mtxAssembly
    EEqn_GPU.checkValue(false);
    //EEqn_GPU.checkValue(true);
    fprintf(stderr, "\n\n=====================EEqn GPUSolver_ part4=====================\n\n\n");

    start1 = std::clock();
    EEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
    fprintf(stderr, "\n\n=====================EEqn GPUSolver_ part5=====================\n\n\n");

    start1 = std::clock();
    EEqn_GPU.updatePsi(&he[0]);
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
    fprintf(stderr, "\n\n=====================EEqn GPUSolver_ part6=====================\n\n\n");
#endif
    fprintf(stderr, "\n\n=====================leave EEqn=====================\n\n\n");
}
