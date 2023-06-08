start = std::clock();
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

#ifdef CPUSolver_
start1 = std::clock();
forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
}
const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();
start1 = std::clock();
time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
time_monitor_YEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

#ifdef GPUSolver_
    start1 = std::clock();
    // // std::thread t(&dfMatrix::solve, &UEqn_GPU);
    UEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

#ifdef GPUSolver_
    start1 = std::clock();
    start2 = std::clock();
    std::vector<double*> Y_old(Y.size()), boundary_Y(Y.size()), boundary_hai(Y.size()), boundary_rhoD(Y.size());
    std::vector<const double*> hai(Y.size()), rhoD(Y.size());
    for (size_t i = 0; i < Y.size(); ++i)
    {
        volScalarField& Yi = Y[i];
        Y_old[i] = &Yi.oldTime()[0];
        cudaMallocHost(&boundary_Y[i], num_boundary_faces*sizeof(double));
        const volScalarField& haii = chemistry->hai(i);
        const volScalarField& rhoDi = chemistry->rhoD(i);
        hai[i] = &haii[0];
        rhoD[i] = &rhoDi[0];
        cudaMallocHost(&boundary_hai[i], num_boundary_faces*sizeof(double));
        cudaMallocHost(&boundary_rhoD[i], num_boundary_faces*sizeof(double));
        int offset = 0;
        forAll(Yi.boundaryField(), patchi)
        {
            const scalarField& patchYi = Yi.boundaryField()[patchi];
            const scalarField& patchHaii = haii.boundaryField()[patchi];
            const scalarField& patchRhoDi = rhoDi.boundaryField()[patchi];
            int patchSize = patchYi.size();

            memcpy(boundary_Y[i] + offset, &patchYi[0], patchSize*sizeof(double));
            memcpy(boundary_hai[i] + offset, &patchHaii[0], patchSize*sizeof(double));
            memcpy(boundary_rhoD[i] + offset, &patchRhoDi[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }

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
    }
    end2 = std::clock();
    time_monitor_YEqn_mtxAssembly_CPU_Prepare += double(end2 - start2) / double(CLOCKS_PER_SEC);

    YEqn_GPU.upwindWeight();
    YEqn_GPU.correctVelocity(Y_old, boundary_Y, hai, boundary_hai, rhoD, boundary_rhoD);
    YEqn_GPU.fvm_ddt();
    YEqn_GPU.fvm_div_phi();
    YEqn_GPU.fvm_div_phiUc();
    YEqn_GPU.fvm_laplacian(&mut_sct[0], boundary_mutsct, rhoD, boundary_rhoD);

    YEqn_GPU.compute_diffAlphaD(&thermo.alpha()[0], hai);
    end1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    YEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_YEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

//MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
label flag_mpi_init;
MPI_Initialized(&flag_mpi_init);
if(flag_mpi_init) MPI_Barrier(PstreamGlobals::MPI_COMM_FOAM);
end = std::clock();
time_monitor_corrDiff += double(end - start) / double(CLOCKS_PER_SEC);

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

    volScalarField Yt(0.0*Y[0]);

    start = std::clock();
    int speciesIndex = 0;
    forAll(Y, i)
    {
        volScalarField& Yi = Y[i];
#ifdef CPUSolver_
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);
#endif
        if (i != inertIndex)
        {
            #ifdef GPUSolver_
                start1 = std::clock();
                YEqn_GPU.updatePsi(&Yi[0], speciesIndex);
                Yi.correctBoundaryConditions();
                end1 = std::clock();
                time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
                time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
                time_monitor_YEqn_mtxAssembly_CPU_Prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);
            #else
                start1 = std::clock();
                tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
                fvScalarMatrix YiEqn
                (
                    fvm::ddt(rho, Yi)
                + mvConvection->fvmDiv(phi, Yi)
                + mvConvection->fvmDiv(phiUc, Yi)
                ==
                    (
                        splitting
                    ?   fvm::laplacian(DEff(), Yi)
                    :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
                    )
                );
                end1 = std::clock();
                time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
                time_monitor_YEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
                // YiEqn.relax();

                start1 = std::clock();
                YiEqn.solve("Yi");
                end1 = std::clock();
                time_monitor_YEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
                time_monitor_YEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
            #endif

            Yi.max(0.0);
            Yt += Yi;
            ++speciesIndex;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    end = std::clock();
    time_monitor_Y += double(end - start) / double(CLOCKS_PER_SEC);
}
