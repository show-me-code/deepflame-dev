{
    volScalarField& he = thermo.he();

#ifdef CPUSolver_
    start1 = std::clock();
    //debug
    // {
    //     const fvPatchScalarField& hew = he.boundaryField()[5];
    //     const basicThermo& bThermo = basicThermo::lookupThermo(hew);
    //     const scalarField& pw = bThermo.p().boundaryField()[5];
    //     fvPatchScalarField& Tw =
    //         const_cast<fvPatchScalarField&>(bThermo.T().boundaryField()[5]);
    //     scalarField& Tw_v = Tw;

    //     Tw.evaluate();

    //     Info << "internal field" <<bThermo.he(pw, Tw, mesh.boundary()[5].faceCells()) << endl;
    //     Info << "boundary field" <<bThermo.he(pw, Tw, 5) << endl;
    //     Info << "calculated grad" << mesh.boundary()[5].deltaCoeffs() * (bThermo.he(pw, Tw, 5) - bThermo.he(pw, Tw, mesh.boundary()[5].faceCells())) << endl;
    // }


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
    tmp<volScalarField> tmpAlpha = turbulence->alphaEff();
    volScalarField alphaEff = tmpAlpha.ref();

    forAll(alphaEff.boundaryField(), patchi){
        labelUList sub_boundary = mesh.boundary()[patchi].faceCells();
        forAll(sub_boundary, i){
            if (sub_boundary[i] == 23){
                printf("patchi = %d, i = %d\n", patchi, i);
                printf("EEqn = %.15lf\n", EEqn.boundaryCoeffs()[patchi][i]);
                printf("gamma_magsf = %.20lf\n", alphaEff.boundaryField()[patchi][i]);
                printf("mesh.magSf() = %.20lf\n", mesh.magSf().boundaryField()[patchi][i]);
                printf("gradientBoundaryCoeffs = %.15lf\n\n", he.boundaryField()[patchi].gradientBoundaryCoeffs()()[i]);
            }
        }
    }

    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // EEqn.relax();
    start1 = std::clock();
    EEqn.solve();
    end1 = std::clock();
    time_monitor_EEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_EEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

#ifdef GPUSolver_
    start1 = std::clock();
    UEqn_GPU.updatePsi(&U[0][0]);
    UEqn_GPU.correctBoundaryConditions();
    U.correctBoundaryConditions();
    K = 0.5*magSqr(U);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

#ifdef GPUSolver_
    // prepare data on CPU
    start1 = std::clock();
    start2 = std::clock();
    // const tmp<volScalarField> alphaEff_tmp(thermo.alpha());
    // const volScalarField& alphaEff = alphaEff_tmp();
    double *alphaEff = nullptr; // tmp
    end2 = std::clock();
    int eeqn_offset = 0;
    int patchNum = 0;

    //debug
    // {
    //     const fvPatchScalarField& hew = he.boundaryField()[5];
    //     const basicThermo& bThermo = basicThermo::lookupThermo(hew);
    //     const scalarField& pw = bThermo.p().boundaryField()[5];
    //     fvPatchScalarField& Tw =
    //         const_cast<fvPatchScalarField&>(bThermo.T().boundaryField()[5]);
    //     scalarField& Tw_v = Tw;

    //     Tw.evaluate();

    //     Info << "internal field" <<bThermo.he(pw, Tw, mesh.boundary()[5].faceCells()) << endl;
    //     Info << "boundary field" <<bThermo.he(pw, Tw, 5) << endl;
    //     Info << "calculated grad" << 
    //         mesh.boundary()[5].deltaCoeffs() * (bThermo.he(pw, Tw, 5) - bThermo.he(pw, Tw, mesh.boundary()[5].faceCells())) << endl;
    // }

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
        // const scalarField& patchAlphaEff = alphaEff.boundaryField()[patchi];
        const scalarField& patchGrad = he.boundaryField()[patchi].gradientBoundaryCoeffs(); // gradient_

        // const DimensionedField<scalar, volMesh>& patchHa_ = he.boundaryField()[patchi];
        // const gradientEnergyFvPatchScalarField patchHa(mesh.boundary()[patchi], patchHa_);
        // const scalarField& patchGrad = patchHa.gradient(); // gradient_
        memcpy(boundary_K + eeqn_offset, &patchK[0], patchSize*sizeof(double));
        // memcpy(boundary_alphaEff + eeqn_offset, &patchAlphaEff[0], patchSize*sizeof(double));
        memcpy(boundary_gradient + eeqn_offset, &patchGradMau[0], patchSize*sizeof(double));

        eeqn_offset += patchSize;

        // debug
        // const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        // // Field<scalar> valueInternalCoeffs = patchFlux*he.boundaryField()[patchi].valueInternalCoeffs(pw);
        // // Field<scalar> valueBoundaryCoeffs = -patchFlux*he.boundaryField()[patchi].valueBoundaryCoeffs(pw);
        // Field<scalar> valueInternalCoeffs = patchFlux*he.boundaryField()[patchi].valueInternalCoeffs(pw);
        // Field<scalar> valueBoundaryCoeffs = -patchFlux*he.boundaryField()[patchi].valueBoundaryCoeffs(pw);
        // Field<scalar> gradientInternalCoeffs = he.boundaryField()[patchi].gradientInternalCoeffs();
        // Field<scalar> gradientBoundaryCoeffs = he.boundaryField()[patchi].gradientBoundaryCoeffs();
        // labelUList sub_boundary = mesh.boundary()[patchi].faceCells();

        // calculate grad manually
        // const fvPatchScalarField& hew = he.boundaryField()[patchi];
        // const basicThermo& bThermo = basicThermo::lookupThermo(hew);
        // const scalarField& ppw = bThermo.p().boundaryField()[patchi];
        // fvPatchScalarField& Tw =
        //     const_cast<fvPatchScalarField&>(bThermo.T().boundaryField()[patchi]);
        // scalarField& Tw_v = Tw;

        // Tw.evaluate();
        // const scalarField& patchDeltaCoeff = mesh.boundary()[patchi].deltaCoeffs();
        // const scalarField heInternal = bThermo.he(ppw, Tw, patchi)();
        // const scalarField heBoundary = bThermo.he(ppw, Tw, mesh.boundary()[patchi].faceCells())();
        // const scalarField patchGradMau = patchDeltaCoeff * (heInternal - heBoundary);
                
        // forAll(sub_boundary, i){
        //     if (sub_boundary[i] == 1)
        //     {
        //         printf("\npatchFlux = %.10lf\n", patchFlux[i]);
        //         printf("valueBoundaryCoeffs = %.10lf\n", he.boundaryField()[patchi].valueBoundaryCoeffs(pw)()[i]);
        //         printf("valueBoundaryCoeffs_CPU = %.10lf\n", valueBoundaryCoeffs[i]);
        //         printf("patchGrad = %.10lf\n", patchGrad[i]);
        //         printf("patchGradMau = %.10lf\n", patchGradMau[i]);
        //         printf("patch = %d, cellID = %d\n", patchi, i);
        //     }
        // }
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
#endif
}
