start = std::clock();
cudaSetDevice(0);
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

upwind<scalar> mvConvectionInterpolation(mesh, phi);
tmp<surfaceScalarField> t_upwind_weight = mvConvectionInterpolation.weights();
surfaceScalarField upwind_weight = t_upwind_weight.ref();

volVectorField sumYGrad
(
    IOobject
    (
        "sumYGrad",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector(dimensionSet(0,-1,0,0,0,0,0), Zero)
);

forAll(Y, i)
{
    sumYDiffError += chemistry->rhoD(i)*fvc::grad(Y[i]);
}
const surfaceScalarField phiUc = linearInterpolate(sumYDiffError) & mesh.Sf();

// for (size_t i = 0; i < num_cells; i++)
// {
//     printf("sumYDiffError[%d] = (%e, %e, %e)\n", i, sumYDiffError[i][0], sumYDiffError[i][1], 
//     sumYDiffError[i][2]);
//     // printf("fvc::grad[%d] = (%e, %e, %e)\n", i, test_grad[i][0], test_grad[i][1], 
//     // test_grad[i][2]);
// }

const surfaceVectorField test_interpolate = linearInterpolate(sumYDiffError).ref();

// for (size_t i = 0; i < num_surfaces; i++)
// {
//     // printf("test_interpolate[%d] = (%e, %e, %e)\n", i, test_interpolate[i][0], test_interpolate[i][1], 
//     // test_interpolate[i][2]);
//     // printf("surface vector[%d] = (%e, %e, %e)\n", i, mesh.Sf()[i][0], mesh.Sf()[i][1], 
//     // mesh.Sf()[i][2]);
//     printf("phiUcOf[%d] = %e\n", i, phiUc[i]);
// }

#ifdef GPUSolver_
    start1 = std::clock();
    // // std::thread t(&dfMatrix::solve, &UEqn_GPU);
    UEqn_GPU.solve();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_Solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

std::vector<double*> Y_new(Y.size()), Y_old(Y.size()), boundary_Y_init(Y.size()), 
    boundary_rhoD_init(Y.size());
std::vector<const double*> rhoD_GPU(Y.size());
for (size_t i = 0; i < Y.size(); ++i)
{
    volScalarField& Yi = Y[i];
    const volScalarField& rhoDi = chemistry->rhoD(i);
    Y_new[i] = &Yi[0];
    Y_old[i] = &Yi.oldTime()[0];
    rhoD_GPU[i] = &chemistry->rhoD(i)[0];
    cudaMallocHost(&boundary_Y_init[i], num_boundary_faces*sizeof(double));
    cudaMallocHost(&boundary_rhoD_init[i], num_boundary_faces*sizeof(double));
    int offset = 0;
    forAll(Yi.boundaryField(), patchi)
    {
        const scalarField& patchYi = Yi.boundaryField()[patchi];
        const scalarField& patchRhoDi = rhoDi.boundaryField()[patchi];
        int patchSize = patchYi.size();

        memcpy(boundary_Y_init[i]+offset, &patchYi[0], patchSize*sizeof(double));
        memcpy(boundary_rhoD_init[i]+offset, &patchRhoDi[0], patchSize*sizeof(double));
        offset += patchSize;
    }
}

volScalarField mut_sct = turbulence->mut().ref()/Sct;
std::vector<double> boundary_sumYDiffError, boundary_phiUc, boundary_mutsct;
forAll(p.boundaryField(), patchi)
{
    const vectorField& patchErri = sumYDiffError.boundaryField()[patchi];
    const scalarField& patchPhiUc = phiUc.boundaryField()[patchi];
    const scalarField& patchMut_sct = mut_sct.boundaryField()[patchi];

    int patchSize = patchErri.size();
    boundary_sumYDiffError.insert(boundary_sumYDiffError.end(), &patchErri[0][0], &patchErri[0][0]+3*patchSize);
    boundary_phiUc.insert(boundary_phiUc.end(), &patchPhiUc[0], &patchPhiUc[0] + patchSize);
    boundary_mutsct.insert(boundary_mutsct.end(), &patchMut_sct[0], &patchMut_sct[0] + patchSize);
}

YEqn_GPU.upwindWeight(&upwind_weight[0]);
YEqn_GPU.correctVelocity(Y_new, boundary_Y_init, rhoD_GPU, &phiUc[0], boundary_sumYDiffError.data(), boundary_phiUc.data());
YEqn_GPU.fvm_ddt(Y_old);
YEqn_GPU.fvm_div_phi();
YEqn_GPU.fvm_div_phiUc();
YEqn_GPU.fvm_laplacian(&mut_sct[0], boundary_mutsct.data(), boundary_rhoD_init);

YEqn_GPU.solve();

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
        hDiffCorrFlux += chemistry->hai(i)*(chemistry->rhoD(i)*fvc::grad(Yi) - Yi*sumYDiffError);
        diffAlphaD += fvc::laplacian(thermo.alpha()*chemistry->hai(i), Yi);

        if (i != inertIndex)
        {
            // tmp<volScalarField> DEff = chemistry->rhoD(i) + turbulence->mut()/Sct;
            // fvScalarMatrix YiEqn
            // (
            //     fvm::ddt(rho, Yi)
            // + mvConvection->fvmDiv(phi, Yi)
            // + mvConvection->fvmDiv(phiUc, Yi)
            // ==
            //     (
            //         splitting
            //     ?   fvm::laplacian(DEff(), Yi)
            //     :  (fvm::laplacian(DEff(), Yi) + combustion->R(Yi))
            //     )
            // );

            // YiEqn.relax();

            // YiEqn.solve("Yi");

            YEqn_GPU.updatePsi(&Yi[0], speciesIndex);
            Yi.correctBoundaryConditions();

            Yi.max(0.0);
            Yt += Yi;
            ++speciesIndex;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);

    end = std::clock();
    time_monitor_Y += double(end - start) / double(CLOCKS_PER_SEC);
    cudaSetDevice(0);
}
