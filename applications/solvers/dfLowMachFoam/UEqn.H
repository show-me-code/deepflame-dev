// #undef GPUSolverNew_
// #define CPUSolver_

// Solve the Momentum equation
#ifdef GPUSolver_
    start1 = std::clock();
    int offset = 0;
    const tmp<volScalarField> nuEff_tmp(turbulence->nuEff());
    const volScalarField& nuEff = nuEff_tmp();
    forAll(U.boundaryField(), patchi)
    {
        const scalarField& patchP = p.boundaryField()[patchi];
        const vectorField& patchU = U.boundaryField()[patchi];
        const scalarField& patchRho = rho.boundaryField()[patchi];
        const scalarField& patchNuEff = nuEff.boundaryField()[patchi];

        int patchSize = patchP.size();

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
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly_CPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    UEqn_GPU.initializeTimeStep();
    U.oldTime();
    UEqn_GPU.fvm_ddt(&U.oldTime()[0][0]);
    UEqn_GPU.fvm_div(boundary_pressure_init, boundary_velocity_init, boundary_nuEff_init, boundary_rho_init);
    UEqn_GPU.fvc_grad(&p[0]);
    UEqn_GPU.fvc_grad_vector();
    UEqn_GPU.dev2T();
    UEqn_GPU.fvc_div_tensor(&nuEff[0]);
    UEqn_GPU.fvm_laplacian();
    UEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_mtxAssembly_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);

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
    // U.oldTime();
    // tmp<fvVectorMatrix> tUEqn
    // (
        // fvm::ddt(rho, U) 
        // + 
        // fvm::div(phi, U)
        // + 
        // turbulence->divDevRhoReff(U) 
        // == -fvc::grad(p)
    // );
    // fvVectorMatrix& UEqn = tUEqn.ref();
    // printf("b_cpu = %e\n", UEqn.source()[1][1]);
    // forAll(U.boundaryField(), patchi){
        // labelUList sub_boundary = mesh.boundary()[patchi].faceCells();
        // forAll(sub_boundary, i){
        //     if (sub_boundary[i] == 1){
        //         printf("b_cpu_bou = %e\n", UEqn.boundaryCoeffs()[patchi][i][1]);
        //         printf("patchi = %d, i = %d\n", patchi, i);
        //     }
        // }
    // }
    // if (pimple.momentumPredictor())
    // {
    //     solve(UEqn);
    //     Info << "U_CPU\n" << U << endl;
    //     K = 0.5*magSqr(U);
    // }
    // UEqn_GPU.checkValue(true);
#elif defined GPUSolverNew_
    const tmp<volScalarField> nuEff_tmp(turbulence->nuEff());
    const volScalarField& nuEff = nuEff_tmp();
    int offset;
    // Info << "nuEff\n" << nuEff << endl;
    // Info << "mu/rho\n" << thermo.mu()/rho << endl;

#if defined DEBUG_
    // run CPU, for temp
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::ddt(rho, U) 
        + 
        fvm::div(phi, U)
        +  
        turbulence->divDevRhoReff(U)
        == -fvc::grad(p)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    volTensorField gradU = fvc::grad(U);

    double *h_boundary_gradU = new double[dfDataBase.num_boundary_surfaces * 9];
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchTensorField& patchGradU = gradU.boundaryField()[patchi];
        int patchsize = patchGradU.size();
        if (patchGradU.type() == "processor") {
            // print info
            if (dynamic_cast<const processorFvPatchField<tensor>&>(patchGradU).doTransform()) {
                Info << "gradU transform = true" << endl;
            } else {
                Info << "gradU transform = false" << endl;
            }
            Info << "rank = " << dynamic_cast<const processorFvPatchField<tensor>&>(patchGradU).rank() << endl;

            memcpy(h_boundary_gradU + 9*offset, &patchGradU[0][0], patchsize * 9 * sizeof(double));
            tensorField patchGradUInternal = 
                    dynamic_cast<const processorFvPatchField<tensor>&>(patchGradU).patchInternalField()();
            memcpy(h_boundary_gradU + 9*offset + patchsize * 9, &patchGradUInternal[0][0], patchsize * 9 * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_gradU + 9*offset, &patchGradU[0][0], patchsize * 9 * sizeof(double));
            offset += patchsize;
        }
    }
#endif

    // run GPU
    // preProcess
    // TODO: preProcessForRhoEqn for temp, now we only transfer phi(instead of rhoEqn) used by fvm::div(phi, U)
    UEqn_GPU.sync();
    
    TICK_START;
    // preparing u, p, nu_eff, and rho.boundary used by UEqn_GPU.preProcess()
    double *h_u = dfDataBase.getFieldPointer("u", location::cpu, position::internal);
    double *h_boundary_u = dfDataBase.getFieldPointer("u", location::cpu, position::boundary);
    double *h_p = dfDataBase.getFieldPointer("p", location::cpu, position::internal);
    double *h_boundary_p = dfDataBase.getFieldPointer("p", location::cpu, position::boundary);
    double *h_nu_eff = UEqn_GPU.getFieldPointer("nu_eff", location::cpu, position::internal);
    double *h_boundary_nu_eff = UEqn_GPU.getFieldPointer("nu_eff", location::cpu, position::boundary);
    TICK_STOP(get pointer);

    TICK_START;
    U.oldTime();
    memcpy(h_nu_eff, &nuEff[0], dfDataBase.cell_value_bytes);
    TICK_STOP(copy to pinned memory);

    TICK_START;
    offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchNuEff = nuEff.boundaryField()[patchi];
        int patchsize = patchNuEff.size();
        if (patchNuEff.type() == "processor") {
            memcpy(h_boundary_nu_eff + offset, &patchNuEff[0], patchsize*sizeof(double));
            memcpy(h_boundary_nu_eff + offset + patchsize, &patchNuEff[0], patchsize*sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_nu_eff + offset, &patchNuEff[0], patchsize*sizeof(double));
            offset += patchsize;
        }
    }
    TICK_STOP(CPU prepare boundary time);

    TICK_START;
    UEqn_GPU.preProcess(h_u, h_boundary_u, h_p, h_boundary_p, h_nu_eff, h_boundary_nu_eff);
    DEBUG_TRACE;
    UEqn_GPU.sync();
    TICK_STOP(GPU preProcess time);

    // process
    TICK_START;
    UEqn_GPU.process();
    TICK_STOP(GPU process time);

    // postProcess
    TICK_START;
    UEqn_GPU.postProcess(h_u);
    memcpy(&U[0][0], h_u, dfDataBase.cell_value_vec_bytes);
    U.correctBoundaryConditions();
    K = 0.5*magSqr(U);
    DEBUG_TRACE;
    TICK_STOP(post process time);

    UEqn_GPU.A(&UEqn_A[0]);
    UEqn_A.correctBoundaryConditions();

#if defined DEBUG_
    // UEqn.relax();
    // UEqn.solve();
    // K = 0.5*magSqr(U);
    // checkResult
    // TODO: for temp, now we compare ldu, finally we compare csr
    std::vector<double> h_internal_coeffs(dfDataBase.num_boundary_surfaces * 3);
    std::vector<double> h_boundary_coeffs(dfDataBase.num_boundary_surfaces * 3);

    offset = 0;
    for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
    {
        int patchsize = dfDataBase.patch_size[patchi];
        const double* internal_coeff_ptr = &UEqn.internalCoeffs()[patchi][0][0];
        const double* boundary_coeff_ptr = &UEqn.boundaryCoeffs()[patchi][0][0];
        memcpy(h_internal_coeffs.data() + offset * 3, internal_coeff_ptr, patchsize * 3 * sizeof(double));
        memcpy(h_boundary_coeffs.data() + offset * 3, boundary_coeff_ptr, patchsize * 3 * sizeof(double));
        offset += patchsize;
    }
    bool printFlag = false;

    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    
    if (!mpi_init_flag || rank == 0) {
        UEqn_GPU.compareResult(&UEqn.lower()[0], &UEqn.upper()[0], &UEqn.diag()[0], &UEqn.source()[0][0],
            h_internal_coeffs.data(), h_boundary_coeffs.data(), 
            // &gradU[0][0], h_boundary_gradU,
            printFlag);
    }
    DEBUG_TRACE;
#endif

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

    UEqn.relax();
    start1 = std::clock();
    if (pimple.momentumPredictor())
    {
        solve(UEqn);

        K = 0.5*magSqr(U);
    }
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

// start1 = std::clock();
// // // std::thread t(&dfMatrix::solve, &UEqn_GPU);
// UEqn_GPU.solve();
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

// start1 = std::clock();
// // // t.join();
// // UEqn_GPU.updatePsi(&U[0][0]);
// K = 0.5*magSqr(U);
// end1 = std::clock();
// time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
// time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
// // Info << "U_amgx = " << U << endl;

// #undef CPUSolver_
// #define GPUSolverNew_
