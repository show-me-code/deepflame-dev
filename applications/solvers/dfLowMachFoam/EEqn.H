{
    volScalarField& he = thermo.he();
#ifdef GPUSolver_
    start1 = std::clock();
    UEqn_GPU.updatePsi(&U[0][0]);
    UEqn_GPU.correctBoundaryConditions();
    U.correctBoundaryConditions();
    K = 0.5*magSqr(U);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // prepare data on CPU
    start1 = std::clock();
    start2 = std::clock();
    // const tmp<volScalarField> alphaEff_tmp(thermo.alpha());
    // const volScalarField& alphaEff = alphaEff_tmp();
    double *alphaEff = nullptr; // tmp
    end2 = std::clock();
    int eeqn_offset = 0;
    int patchNum = 0;

    forAll(he.boundaryField(), patchi)
    {
        patchNum++;
        const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        int patchSize = pw.size();

        // construct gradient manually
        const fvPatchScalarField& hew = he.boundaryField()[patchi];
        const basicThermo& bThermo = basicThermo::lookupThermo(hew);
        const scalarField& ppw = bThermo.p().boundaryField()[patchi];
        fvPatchScalarField& Tw =
            const_cast<fvPatchScalarField&>(bThermo.T().boundaryField()[patchi]);
        scalarField& Tw_v = Tw;

        Tw.evaluate();
        const scalarField& patchDeltaCoeff = mesh.boundary()[patchi].deltaCoeffs();
        const scalarField heInternal = bThermo.he(ppw, Tw, patchi)();
        const scalarField heBoundary = bThermo.he(ppw, Tw, mesh.boundary()[patchi].faceCells())();
        const scalarField patchGradMau = patchDeltaCoeff * (heInternal - heBoundary);

        const scalarField& patchK = K.boundaryField()[patchi];
        // const scalarField& patchAlphaEff = alphaEff.boundaryField()[patchi]; // not H2Dcopy when use UnityLewis
        // const scalarField& patchGrad = he.boundaryField()[patchi].gradientBoundaryCoeffs(); // gradient_

        // const DimensionedField<scalar, volMesh>& patchHa_ = he.boundaryField()[patchi];
        // const gradientEnergyFvPatchScalarField patchHa(mesh.boundary()[patchi], patchHa_);
        // const scalarField& patchGrad = patchHa.gradient(); // gradient_
        memcpy(boundary_K + eeqn_offset, &patchK[0], patchSize*sizeof(double));
        // memcpy(boundary_alphaEff + eeqn_offset, &patchAlphaEff[0], patchSize*sizeof(double));
        memcpy(boundary_gradient + eeqn_offset, &patchGradMau[0], patchSize*sizeof(double));

        eeqn_offset += patchSize;
    }
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly_CPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);
    fprintf(stderr, "time_monitor_EEqn_mtxAssembly_CPU_prepare: %lf, build alphaEff time: %lf, patchNum: %d\n",
            time_monitor_EEqn_mtxAssembly_CPU_prepare,
            double(end2 - start2) / double(CLOCKS_PER_SEC), patchNum);

    // prepare data on GPU
    start1 = std::clock();
    he.oldTime();
    K.oldTime();
    EEqn_GPU.prepare_data(&he.oldTime()[0], &K[0], &K.oldTime()[0], alphaEff,
            &dpdt[0], boundary_K, boundary_alphaEff, boundary_gradient);
    EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly_GPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    EEqn_GPU.initializeTimeStep();
    EEqn_GPU.fvm_ddt();
    EEqn_GPU.fvm_div();
    EEqn_GPU.fvm_laplacian();
    EEqn_GPU.fvc_ddt();
    EEqn_GPU.fvc_div_phi_scalar();
    EEqn_GPU.fvc_div_vector();
    EEqn_GPU.add_to_source();
    EEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // check value of mtxAssembly, no time monitor
    // EEqn_GPU.checkValue(true);

    start1 = std::clock();
    EEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    EEqn_GPU.updatePsi(&he[0]);
    he.correctBoundaryConditions();
    he.write();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#elif defined GPUSolverNew_
    double *h_he = dfDataBase.getFieldPointer("he", location::cpu, position::internal);
    double *h_k = dfDataBase.getFieldPointer("k", location::cpu, position::internal);
    double *h_k_old = dfDataBase.getFieldPointer("k_old", location::cpu, position::internal);
    double *h_boundary_k = dfDataBase.getFieldPointer("k", location::cpu, position::boundary);
    double *h_boundary_heGradient = EEqn_GPU.getFieldPointer("heGradient", location::cpu, position::boundary);
    double *h_dpdt = EEqn_GPU.getFieldPointer("dpdt", location::cpu, position::internal);
    he.oldTime();
    K.oldTime();
    memcpy(h_he, &he.oldTime()[0], dfDataBase.cell_value_bytes); 
    memcpy(h_k, &K[0], dfDataBase.cell_value_bytes); 
    memcpy(h_k_old, &K.oldTime()[0], dfDataBase.cell_value_bytes); 
    memcpy(h_dpdt, &dpdt[0], dfDataBase.cell_value_bytes); 

    // get boundary value
    int eeqn_offset = 0;
    int patchNum = 0;
    //TODO: check the difference between patchGradMau and patchGrad
    forAll(he.boundaryField(), patchi)
    {
        patchNum++;
        const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        int patchSize = pw.size();

        // construct gradient manually
        const fvPatchScalarField& hew = he.boundaryField()[patchi];
        const basicThermo& bThermo = basicThermo::lookupThermo(hew);
        const scalarField& ppw = bThermo.p().boundaryField()[patchi];
        fvPatchScalarField& Tw =
            const_cast<fvPatchScalarField&>(bThermo.T().boundaryField()[patchi]);
        scalarField& Tw_v = Tw;

        Tw.evaluate();
        const scalarField& patchDeltaCoeff = mesh.boundary()[patchi].deltaCoeffs();
        const scalarField heInternal = bThermo.he(ppw, Tw, patchi)();
        const scalarField heBoundary = bThermo.he(ppw, Tw, mesh.boundary()[patchi].faceCells())();
        const scalarField patchGradMau = patchDeltaCoeff * (heInternal - heBoundary);

        const scalarField& patchK = K.boundaryField()[patchi];
        // const scalarField& patchAlphaEff = alphaEff.boundaryField()[patchi]; // not H2Dcopy when use UnityLewis
        // const scalarField& patchGrad = he.boundaryField()[patchi].gradientBoundaryCoeffs(); // gradient_

        // const DimensionedField<scalar, volMesh>& patchHa_ = he.boundaryField()[patchi];
        // const gradientEnergyFvPatchScalarField patchHa(mesh.boundary()[patchi], patchHa_);
        // const scalarField& patchGrad = patchHa.gradient(); // gradient_
        memcpy(h_boundary_k + eeqn_offset, &patchK[0], patchSize*sizeof(double)); // tmp
        // memcpy(boundary_alphaEff + eeqn_offset, &patchAlphaEff[0], patchSize*sizeof(double));
        memcpy(h_boundary_heGradient + eeqn_offset, &patchGradMau[0], patchSize*sizeof(double)); // tmp

        eeqn_offset += patchSize;
    }

    EEqn_GPU.preProcess(h_he, h_k, h_k_old, h_dpdt, h_boundary_k, h_boundary_heGradient);

    EEqn_GPU.process();

    EEqn_GPU.postProcess(h_he);
    memcpy(&he[0], h_he, dfDataBase.cell_value_bytes);
    he.correctBoundaryConditions();

#if defined DEBUG_
    fvScalarMatrix EEqn
    (

        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    ==
        (
            turbName == "laminar"
            ?
            (
                fvm::laplacian(turbulence->alpha(), he)
            -   diffAlphaD
            +   fvc::div(hDiffCorrFlux)
            )
            :
            (
                fvm::laplacian(turbulence->alphaEff(), he)
            )
        )
    );
    // EEqn.relax();
    // EEqn.solve("ha");
    // checkResult
    // TODO: for temp, now we compare ldu, finally we compare csr
    std::vector<double> h_internal_coeffs(dfDataBase.num_boundary_surfaces);
    std::vector<double> h_boundary_coeffs(dfDataBase.num_boundary_surfaces);

    offset = 0;
    for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
    {
        int patchsize = dfDataBase.patch_size[patchi];
        const double* internal_coeff_ptr = &EEqn.internalCoeffs()[patchi][0];
        const double* boundary_coeff_ptr = &EEqn.boundaryCoeffs()[patchi][0];
        memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchsize * sizeof(double));
        memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchsize * sizeof(double));
        offset += patchsize;
    }
    bool printFlag = false;
    EEqn_GPU.compareResult(&EEqn.lower()[0], &EEqn.upper()[0], &EEqn.diag()[0], &EEqn.source()[0],
            h_internal_coeffs.data(), h_boundary_coeffs.data(), printFlag);
    DEBUG_TRACE;
#endif

#else
    start1 = std::clock();
    fvScalarMatrix EEqn
    (

        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
    +   fvc::ddt(rho, K) + fvc::div(phi, K)
    -   dpdt
    ==
        (
            turbName == "laminar"
            ?
            (
                fvm::laplacian(turbulence->alpha(), he)
            -   diffAlphaD
            +   fvc::div(hDiffCorrFlux)
            )
            :
            (
                fvm::laplacian(turbulence->alphaEff(), he)
            )
        )
    );
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    EEqn.relax();
    start1 = std::clock();
    EEqn.solve("ha");
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif
}
