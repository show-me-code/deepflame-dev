// #undef GPUSolverNew_
// #define CPUSolver_
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
    int patchNum = 0;

    offset = 0;
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
        memcpy(boundary_K + offset, &patchK[0], patchSize*sizeof(double));
        // memcpy(boundary_alphaEff + offset, &patchAlphaEff[0], patchSize*sizeof(double));
        memcpy(boundary_gradient + offset, &patchGradMau[0], patchSize*sizeof(double));

        offset += patchSize;
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
    double *h_boundary_he = dfDataBase.getFieldPointer("he", location::cpu, position::boundary);
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
    //TODO: check the difference between patchGradMau and patchGrad
    offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchK = K.boundaryField()[patchi];
        int patchSize = patchK.size();
          if (patchK.type() == "processor") {
              scalarField patchKInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchK).patchInternalField()();
              memcpy(h_boundary_k + offset, &patchK[0], patchSize * sizeof(double));
              memcpy(h_boundary_k + offset + patchSize,
                      &patchKInternal[0], patchSize * sizeof(double));
              offset += 2 * patchSize;
          } else {
            memcpy(h_boundary_k + offset, &patchK[0], patchSize*sizeof(double));

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

            // const scalarField& patchAlphaEff = alphaEff.boundaryField()[patchi]; // not H2Dcopy when use UnityLewis
            // const scalarField& patchGrad = he.boundaryField()[patchi].gradientBoundaryCoeffs(); // gradient_

            // const DimensionedField<scalar, volMesh>& patchHa_ = he.boundaryField()[patchi];
            // const gradientEnergyFvPatchScalarField patchHa(mesh.boundary()[patchi], patchHa_);
            // const scalarField& patchGrad = patchHa.gradient(); // gradient_
            // memcpy(boundary_alphaEff + offset, &patchAlphaEff[0], patchSize*sizeof(double));
            memcpy(h_boundary_heGradient + offset, &patchGradMau[0], patchSize*sizeof(double)); // tmp

            offset += patchSize;
        }
    }

    EEqn_GPU.preProcess(h_he, h_k, h_k_old, h_dpdt, h_boundary_k, h_boundary_heGradient);

    DEBUG_TRACE;
    EEqn_GPU.process();

    DEBUG_TRACE;
    EEqn_GPU.postProcess(h_he, h_boundary_he);
    DEBUG_TRACE;

    // copy h_he to he(cpu)
    // memcpy(&he[0], h_he, dfDataBase.cell_value_bytes);

    //DEBUG_TRACE;
    //he.correctBoundaryConditions();
    //DEBUG_TRACE;

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
    EEqn.solve("ha");
    // checkResult
    // TODO: for temp, now we compare ldu, finally we compare csr
    std::vector<double> h_internal_coeffs(dfDataBase.num_boundary_surfaces);
    std::vector<double> h_boundary_coeffs(dfDataBase.num_boundary_surfaces);

    offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchHe = he.boundaryField()[patchi];
        int patchSize = patchHe.size();
        const double* internal_coeff_ptr = &EEqn.internalCoeffs()[patchi][0];
        const double* boundary_coeff_ptr = &EEqn.boundaryCoeffs()[patchi][0];
        if (patchHe.type() == "processor") {
            memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchSize * sizeof(double));
            memset(h_internal_coeffs.data() + offset + patchSize, 0, patchSize * sizeof(double));
            memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchSize * sizeof(double));
            memset(h_boundary_coeffs.data() + offset + patchSize, 0, patchSize * sizeof(double));
            offset += patchSize * 2;
        } else {
            memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchSize * sizeof(double));
            memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchSize * sizeof(double));
            offset += patchSize;
        }
    }

    double *h_boundary_he_tmp = new double[dfDataBase.num_boundary_surfaces];
    offset = 0;
    forAll(he.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchHe = he.boundaryField()[patchi];
        int patchSize = patchHe.size();
        if (patchHe.type() == "processor") {
            const scalarField& patchHeInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchHe).patchInternalField()();
            memcpy(h_boundary_he_tmp + offset, &patchHe[0], patchSize * sizeof(double));
            memcpy(h_boundary_he_tmp + offset + patchSize, &patchHeInternal[0], patchSize * sizeof(double));
            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_he_tmp + offset, &patchHe[0], patchSize * sizeof(double));
            offset += patchSize;
        }
    }

    bool printFlag = false;
    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (!mpi_init_flag || rank == 0) {
		// DEBUG_TRACE;
		// EEqn_GPU.compareResult(&EEqn.lower()[0], &EEqn.upper()[0], &EEqn.diag()[0], &EEqn.source()[0],
		// 		h_internal_coeffs.data(), h_boundary_coeffs.data(), printFlag);
		// DEBUG_TRACE;
        EEqn_GPU.compareHe(&he[0], h_boundary_he_tmp, printFlag);
    }

    delete h_boundary_he_tmp;

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
// #undef CPUSolver_
// #define GPUSolverNew_
