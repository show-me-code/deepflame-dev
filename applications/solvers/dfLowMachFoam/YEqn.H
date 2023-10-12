// #undef GPUSolverNew_
// #define CPUSolver_
hDiffCorrFlux = Zero;
diffAlphaD = Zero;
sumYDiffError = Zero;

tmp<fv::convectionScheme<scalar>> mvConvection
(
    fv::convectionScheme<scalar>::New
    (
        mesh,
        fields,
        phi,
        mesh.divScheme("div(phi,Yi_h)")
    )
);
#if defined GPUSolverNew_
#if defined DEBUG_
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(0)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(1)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(2)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(3)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(4)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(5)));
    //randomInitField<volScalarField>(const_cast<volScalarField&>(chemistry->hai(6)));
#endif
#endif

#ifdef GPUSolver_
    start1 = std::clock();
    UEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    std::vector<double*> Y_old(Y.size()), boundary_Y(Y.size()), boundary_hai(Y.size()), boundary_rhoD(Y.size());
    std::vector<const double*> hai(Y.size()), rhoD(Y.size());
    for (size_t i = 0; i < Y.size(); ++i)
    {
        volScalarField& Yi = Y[i];
        Yi.oldTime();
        Y_old[i] = &Yi.oldTime()[0];
        if (updateBoundaryFields)
        {
            cudaMallocHost(&boundary_Y[i], num_boundary_faces*sizeof(double));
        }
        const volScalarField& haii = chemistry->hai(i);
        const volScalarField& rhoDi = chemistry->rhoD(i);
        // hai[i] = &haii[0];
        rhoD[i] = &rhoDi[0];
        // cudaMallocHost(&boundary_hai[i], num_boundary_faces*sizeof(double));
        cudaMallocHost(&boundary_rhoD[i], num_boundary_faces*sizeof(double));
        int offset = 0;
        forAll(Yi.boundaryField(), patchi)
        {
            const scalarField& patchYi = Yi.boundaryField()[patchi];
            // const scalarField& patchHaii = haii.boundaryField()[patchi];
            const scalarField& patchRhoDi = rhoDi.boundaryField()[patchi];
            int patchSize = patchYi.size();

            if (updateBoundaryFields)
            {
                memcpy(boundary_Y[i] + offset, &patchYi[0], patchSize*sizeof(double));
            }
            // memcpy(boundary_hai[i] + offset, &patchHaii[0], patchSize*sizeof(double));
            memcpy(boundary_rhoD[i] + offset, &patchRhoDi[0], patchSize*sizeof(double));
            offset += patchSize;
        }
        // if (i == 5)
        // {
        //     Info << "rhoD_CPU" << rhoDi << endl;
        // }
        
    }
    // Info << "rhoD from nuEff\n" << nuEff * rho / 0.7 << endl;
    updateBoundaryFields = false;
    volScalarField mut_sct = turbulence->mut().ref()/Sct;
    double *boundary_mutsct = nullptr;
    cudaMallocHost(&boundary_mutsct, num_boundary_faces*sizeof(double));
    int offset = 0;
    forAll(p.boundaryField(), patchi)
    {
        const scalarField& patchMut_sct = mut_sct.boundaryField()[patchi];
        int patchSize = patchMut_sct.size();
        memcpy(boundary_mutsct + offset, &patchMut_sct[0], patchSize*sizeof(double));
        offset += patchSize;

        // debug
        // const fvsPatchScalarField& pw = mesh.surfaceInterpolation::weights().boundaryField()[patchi];
        // Field<scalar> valueInternalCoeffs = Y[5].boundaryField()[patchi].valueInternalCoeffs(pw);
        // Field<scalar> valueBoundaryCoeffs = Y[5].boundaryField()[patchi].valueBoundaryCoeffs(pw);
        // Field<scalar> gradientInternalCoeffs = Y[5].boundaryField()[patchi].gradientInternalCoeffs();
        // Field<scalar> gradientBoundaryCoeffs = Y[5].boundaryField()[patchi].gradientBoundaryCoeffs();
        // Info << "valueInternalCoeffs\n" << valueInternalCoeffs << endl;
        // Info << "valueBoundaryCoeffs\n" << valueBoundaryCoeffs << endl;
        // Info << "gradientInternalCoeffs\n" << gradientInternalCoeffs << endl;
        // Info << "gradientBoundaryCoeffs\n" << gradientBoundaryCoeffs << endl;
    }
    end1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_mtxAssembly_CPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);
    //fprintf(stderr, "time_monitor_YEqn_mtxAssembly_CPU_prepare: %lf\n", time_monitor_YEqn_mtxAssembly_CPU_prepare);

    start1 = std::clock();
    YEqn_GPU.initializeTimeStep();
    YEqn_GPU.upwindWeight();
    YEqn_GPU.fvm_laplacian_and_sumYDiffError_diffAlphaD_hDiffCorrFlux(Y_old, boundary_Y,
            hai, boundary_hai, rhoD, boundary_rhoD, &mut_sct[0], boundary_mutsct, &thermo.alpha()[0]);
    YEqn_GPU.fvm_ddt();
    YEqn_GPU.fvm_div_phi();
    YEqn_GPU.fvm_div_phiUc();
    YEqn_GPU.sync();
    // YEqn_GPU.checkValue(true, "of_output_H2.txt");
    end1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_mtxAssembly_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);
    //fprintf(stderr, "time_monitor_YEqn_mtxAssembly_GPU_run: %lf\n", time_monitor_YEqn_mtxAssembly_GPU_run);

    start1 = std::clock();
    YEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#elif defined GPUSolverNew_
#if defined DEBUG_
    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    }
    const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();
#else
    // do nothing
#endif
#else // should only for CPUSolver
    start1 = std::clock();
    forAll(Y, i)
    {
        sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
    }
    // Info << "sumYDiffError\n" << sumYDiffError << endl;
    const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();
    start1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);  
#endif

//MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
label flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);

{
    if (!splitting)
    {
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        combustion->correct();
        //label flag_mpi_init;
        //MPI_Initialized(&flag_mpi_init);
        if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
        std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
        time_monitor_chem += processingTime.count();
    }

#ifdef GPUSolver_
    start1 = std::clock();
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        YEqn_GPU.updatePsi(&Yi[0], i);
        Yi.correctBoundaryConditions();
    } 
    YEqn_GPU.correctBoundaryConditions();
    end1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#elif defined GPUSolverNew_
#if defined DEBUG_
    // run CPU
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
    }

    int specie_index = 0;

    // should compute grad_yi before YiEqn.solve()
    const volVectorField grad_yi = fvc::grad(Y[specie_index]);

    tmp<volScalarField> DEff = chemistry->rhoD(specie_index) + turbulence->mut()/Sct;
    fvScalarMatrix YiEqn
        (
         fvm::ddt(rho, Y[specie_index])
         + mvConvection->fvmDiv(phi, Y[specie_index])
         + mvConvection->fvmDiv(phiUc, Y[specie_index])
         ==
         fvm::laplacian(DEff(), Y[specie_index])
        );
    // YiEqn.relax();
    // YiEqn.solve("Yi");
    Y[specie_index].max(0.0);
#endif

    TICK_START;
    // run GPU
    // preProcess
    // preparing rhoD, hai, mut_sct used by YEqn_GPU.preProcess()
    double *h_rhoD = YEqn_GPU.getFieldPointer("rhoD", location::cpu, position::internal);
    double *h_boundary_rhoD = YEqn_GPU.getFieldPointer("rhoD", location::cpu, position::boundary);
    // UnityLewis
    double *h_hai = YEqn_GPU.getFieldPointer("hai", location::cpu, position::internal);
    double *h_boundary_hai = YEqn_GPU.getFieldPointer("hai", location::cpu, position::boundary);
    for (size_t i = 0; i < Y.size(); ++i)
    {
        const volScalarField& rhoDi = chemistry->rhoD(i);
        memcpy(h_rhoD + dfDataBase.num_cells * i, &rhoDi[0], dfDataBase.num_cells * sizeof(double));
        // UnityLewis
        const volScalarField& haii = chemistry->hai(i);
        memcpy(h_hai + dfDataBase.num_cells * i, &haii[0], dfDataBase.num_cells * sizeof(double));
        int offset = 0;
        forAll(rhoDi.boundaryField(), patchi)
        {
            const fvPatchScalarField& patchRhoDi = rhoDi.boundaryField()[patchi];
            const fvPatchScalarField& patchHaii = haii.boundaryField()[patchi];
            int patchSize = patchRhoDi.size();

            if (patchRhoDi.type() == "processor") {
                int rank = dynamic_cast<const processorFvPatchField<scalar>&>(patchRhoDi).rank();
                if (dynamic_cast<const processorFvPatchField<scalar>&>(patchRhoDi).doTransform()) {
                    Info << "rank = " << rank << ", rhoD transform = true" << endl;
                } else {
                    Info << "rank = " << rank << ", rhoD transform = false" << endl;
                }

                scalarField patchRhoDiInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchRhoDi).patchInternalField()();
                memcpy(h_boundary_rhoD + dfDataBase.num_boundary_surfaces * i + offset, &patchRhoDi[0], patchSize * sizeof(double));
                memcpy(h_boundary_rhoD + dfDataBase.num_boundary_surfaces * i + offset + patchSize,
                    &patchRhoDiInternal[0], patchSize * sizeof(double));

                // UnityLewis
                scalarField patchHaiiInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchHaii).patchInternalField()();
                memcpy(h_boundary_hai + dfDataBase.num_boundary_surfaces * i + offset, &patchHaii[0], patchSize*sizeof(double));
                memcpy(h_boundary_hai + dfDataBase.num_boundary_surfaces * i + offset + patchSize,
                    &patchHaiiInternal[0], patchSize*sizeof(double));

                offset += patchSize * 2;
            } else {
                memcpy(h_boundary_rhoD + dfDataBase.num_boundary_surfaces * i + offset, &patchRhoDi[0], patchSize * sizeof(double));
                memcpy(h_boundary_hai + dfDataBase.num_boundary_surfaces * i + offset, &patchHaii[0], patchSize*sizeof(double));
                offset += patchSize;
            }
        }
    }
    // UnityLewis
    volScalarField mut_sct = turbulence->mut().ref() / Sct;
    double *h_mut_sct = YEqn_GPU.getFieldPointer("mut_sct", location::cpu, position::internal);
    double *h_boundary_mut_sct = YEqn_GPU.getFieldPointer("mut_sct", location::cpu, position::boundary);
    memcpy(h_mut_sct, &mut_sct[0], dfDataBase.num_cells * sizeof(double));
    int offset = 0;
    forAll(mut_sct.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchMutSct = mut_sct.boundaryField()[patchi];
        int patchSize = patchMutSct.size();
        if (patchMutSct.type() == "processor") {
            int rank = dynamic_cast<const processorFvPatchField<scalar>&>(patchMutSct).rank();
            if (dynamic_cast<const processorFvPatchField<scalar>&>(patchMutSct).doTransform()) {
                Info << "rank = " << rank << ", mut_sct transform = true" << endl;
            } else {
                Info << "rank = " << rank << ", mut_sct transform = false" << endl;
            }
            scalarField patchMutSctInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchMutSct).patchInternalField()();
            memcpy(h_boundary_mut_sct + offset, &patchMutSct[0], patchSize*sizeof(double));
            memcpy(h_boundary_mut_sct + offset + patchSize,
                    &patchMutSctInternal[0], patchSize*sizeof(double));
            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_mut_sct + offset, &patchMutSct[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }
    TICK_STOP(prepare on CPU);

    TICK_START;
    YEqn_GPU.preProcess(h_rhoD, h_boundary_rhoD, h_hai, h_boundary_hai, h_mut_sct, h_boundary_mut_sct);
    TICK_STOP(prepare on GPU);

    DEBUG_TRACE;
    TICK_START;
    // process
    YEqn_GPU.process();
    TICK_STOP(process on GPU);
    DEBUG_TRACE;

#if defined DEBUG_
    std::vector<double> h_boundary_diffAlphaD;
    std::vector<double> h_boundary_grad_yi;
    std::vector<double> h_boundary_sumYDiffError;
    std::vector<double> h_boundary_hDiffCorrFlux;
    std::vector<double> h_boundary_phiUc;
    h_boundary_diffAlphaD.resize(dfDataBase.num_boundary_surfaces);
    h_boundary_grad_yi.resize(dfDataBase.num_boundary_surfaces * 3);
    h_boundary_sumYDiffError.resize(dfDataBase.num_boundary_surfaces * 3);
    h_boundary_hDiffCorrFlux.resize(dfDataBase.num_boundary_surfaces * 3);
    h_boundary_phiUc.resize(dfDataBase.num_boundary_surfaces);
    offset = 0;
    forAll(diffAlphaD.boundaryField(), patchi)
    {
        //const scalarField& patchdiffAlphaD = diffAlphaD.boundaryField()[patchi];
        const fvPatchScalarField& patchdiffAlphaD = diffAlphaD.boundaryField()[patchi];
        const fvPatchVectorField& patchgradyi = grad_yi.boundaryField()[patchi];
        const fvPatchVectorField& patchsumYDiffError = sumYDiffError.boundaryField()[patchi];
        const fvPatchVectorField& patchhDiffCorrFlux = hDiffCorrFlux.boundaryField()[patchi];
        const fvsPatchScalarField& patchphiUc = phiUc.boundaryField()[patchi];
        //fprintf(stderr, "myRank[%d] patchphiUc type: %s\n", myRank, patchphiUc.type().c_str());
        int patchSize = patchdiffAlphaD.size();
        if (patchdiffAlphaD.type() == "processor") {
            scalarField patchdiffAlphaDInternal = dynamic_cast<const processorFvPatchField<scalar>&>(patchdiffAlphaD).patchInternalField()();
            vectorField patchgradyiInternal = dynamic_cast<const processorFvPatchField<vector>&>(patchgradyi).patchInternalField()();
            vectorField patchsumYDiffErrorInternal = dynamic_cast<const processorFvPatchField<vector>&>(patchsumYDiffError).patchInternalField()();
            vectorField patchhDiffCorrFluxInternal = dynamic_cast<const processorFvPatchField<vector>&>(patchhDiffCorrFlux).patchInternalField()();
            memcpy(h_boundary_diffAlphaD.data() + offset, &patchdiffAlphaD[0], patchSize*sizeof(double));
            memcpy(h_boundary_diffAlphaD.data() + offset + patchSize, &patchdiffAlphaDInternal[0], patchSize*sizeof(double));
            memcpy(h_boundary_grad_yi.data() + offset * 3, &patchgradyi[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_grad_yi.data() + (offset + patchSize) * 3, &patchgradyiInternal[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_sumYDiffError.data() + offset * 3, &patchsumYDiffError[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_sumYDiffError.data() + (offset + patchSize) * 3, &patchsumYDiffErrorInternal[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_hDiffCorrFlux.data() + offset * 3, &patchhDiffCorrFlux[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_hDiffCorrFlux.data() + (offset + patchSize) * 3, &patchhDiffCorrFluxInternal[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_phiUc.data() + offset, &patchphiUc[0], patchSize*sizeof(double));
            memcpy(h_boundary_phiUc.data() + offset, &patchphiUc[0], patchSize*sizeof(double));
            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_diffAlphaD.data() + offset, &patchdiffAlphaD[0], patchSize*sizeof(double));
            memcpy(h_boundary_grad_yi.data() + offset * 3, &patchgradyi[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_sumYDiffError.data() + offset * 3, &patchsumYDiffError[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_hDiffCorrFlux.data() + offset * 3, &patchhDiffCorrFlux[0][0], patchSize * 3 *sizeof(double));
            memcpy(h_boundary_phiUc.data() + offset, &patchphiUc[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }
    DEBUG_TRACE;
    YEqn_GPU.comparediffAlphaD(&diffAlphaD[0], h_boundary_diffAlphaD.data(), false);
    YEqn_GPU.comparegradyi(&grad_yi[0][0], h_boundary_grad_yi.data(), specie_index, false);
    YEqn_GPU.comparesumYDiffError(&sumYDiffError[0][0], h_boundary_sumYDiffError.data(), false);
    YEqn_GPU.comparehDiffCorrFlux(&hDiffCorrFlux[0][0], h_boundary_hDiffCorrFlux.data(), false);
    YEqn_GPU.comparephiUc(&phiUc[0], h_boundary_phiUc.data(), false);
    DEBUG_TRACE;

    // checkResult
    // TODO: for temp, now we compare ldu, finally we compare csr
    std::vector<double> yeqn_h_internal_coeffs(dfDataBase.num_boundary_surfaces);
    std::vector<double> yeqn_h_boundary_coeffs(dfDataBase.num_boundary_surfaces);
    offset = 0;
    for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
    {
        int patchsize = dfDataBase.patch_size[patchi];
        const double* internal_coeff_ptr = &YiEqn.internalCoeffs()[patchi][0];
        const double* boundary_coeff_ptr = &YiEqn.boundaryCoeffs()[patchi][0];
        memcpy(yeqn_h_internal_coeffs.data() + offset, internal_coeff_ptr, patchsize * sizeof(double));
        memcpy(yeqn_h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchsize * sizeof(double));
        offset += patchsize;
    }
    // NOTE: ldu and yi can't be compared at the same time
    // to compare ldu data, you should open both DEBUG_ and DEBUG_CHECK_LDU in src_gpu
    // to compare yi, you should only open DEBUG_ in src_gpu.
    // Besides, if you compare ldu data, be patient to keep specie_index in YEqn.H and dfYEqn.cu the same.
    //DEBUG_TRACE;
    YEqn_GPU.compareResult(&YiEqn.lower()[0], &YiEqn.upper()[0], &YiEqn.diag()[0], &YiEqn.source()[0],
       yeqn_h_internal_coeffs.data(), yeqn_h_boundary_coeffs.data(), false);

    DEBUG_TRACE;
    // YEqn_GPU.compareYi(&Y[specie_index][0], specie_index, false);
    // DEBUG_TRACE;
#endif

    TICK_START;
    // postProcess
    double *h_y = dfDataBase.getFieldPointer("y", location::cpu, position::internal);
    double *h_boundary_y = dfDataBase.getFieldPointer("y", location::cpu, position::boundary);
    YEqn_GPU.postProcess(h_y, h_boundary_y);
    TICK_STOP(postProcess on GPU);
    DEBUG_TRACE;

    // copy h_y to Y(cpu)
    offset = 0;
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        memcpy(&Yi[0], h_y + offset, Yi.size() * sizeof(double));
        offset += Yi.size();
    }

    // copy h_boundary_y to Y(cpu)
    offset = 0;
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        forAll(Yi.boundaryField(), patchi) {
            const fvPatchScalarField& const_patchYi = Yi.boundaryField()[patchi];
            fvPatchScalarField& patchYi = const_cast<fvPatchScalarField&>(const_patchYi);
            int patchsize = patchYi.size();
            if (patchYi.type() == "processor") {
                const scalarField const_patchYiInternal =
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchYi).patchInternalField()();
                scalarField patchYiInternal = const_cast<scalarField&>(const_patchYiInternal);
                memcpy(&patchYi[0], h_boundary_y + offset, patchsize * sizeof(double));
                memcpy(&patchYiInternal[0], h_boundary_y + offset + patchsize, patchsize * sizeof(double));
                offset += patchsize * 2;
            } else {
                memcpy(&patchYi[0], h_boundary_y + offset, patchsize * sizeof(double));
                offset += patchsize;
            }
        }
    }
    DEBUG_TRACE;

    fflush(stderr);
#else
    start2 = std::clock();
    volScalarField Yt(0.0*Y[0]);
    int speciesIndex = 0;
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
        if (i != inertIndex)
        {
            start1 = std::clock();
            tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;

            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
            +
                (
                    turbName == "laminar"
                    ?  (mvConvection->fvmDiv(phi, Yi) + mvConvection->fvmDiv(phiUc, Yi))
                    :   mvConvection->fvmDiv(phi, Yi)
                )
            ==
                (
                    splitting
                    ?   fvm::laplacian(DEff(), Yi)
                    :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
                    )
            );
            
            end1 = std::clock();
            time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
            YiEqn.relax();

            start1 = std::clock();
            YiEqn.solve("Yi");
            end1 = std::clock();
            time_monitor_YEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

            Yi.max(0.0);
            Yt += Yi;
            ++speciesIndex;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
    end2 = std::clock();
    time_monitor_YEqn += double(end2 - start2) / double(CLOCKS_PER_SEC);
#endif
}
// #undef CPUSolver_
// #define GPUSolverNew_
