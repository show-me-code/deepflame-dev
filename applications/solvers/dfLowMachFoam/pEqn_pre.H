// #undef GPUSolverNew_
// #define CPUSolver_
if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

#if defined GPUSolverNew_
// tmp: correct rho
double *h_boundary_rho_p = new double[dfDataBase.num_boundary_surfaces];
offset = 0;
forAll(rho.boundaryField(), patchi)
{
    const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
    int patchsize = patchRho.size();
    if (patchRho.type() == "processor") {
        memcpy(h_boundary_rho_p + offset, &patchRho[0], patchsize * sizeof(double));
        scalarField patchRhoInternal = 
                dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).patchInternalField()();
        memcpy(h_boundary_rho_p + offset + patchsize, &patchRhoInternal[0], patchsize * sizeof(double));
        offset += patchsize * 2;
    } else {
        memcpy(h_boundary_rho_p + offset, &patchRho[0], patchsize * sizeof(double));
        offset += patchsize;
    }
}
rhoEqn_GPU.correctPsi(&rho[0], h_boundary_rho_p);

delete[] h_boundary_rho_p;

DEBUG_TRACE;

// tmp: correct phi
double *h_boundary_phi_p = new double[dfDataBase.num_boundary_surfaces];
offset = 0;
forAll(phi.boundaryField(), patchi)
{
    const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
    int patchsize = patchPhi.size();

    if (patchPhi.type() == "processor") {
        memcpy(h_boundary_phi_p + offset, &patchPhi[0], patchsize * sizeof(double));
        memcpy(h_boundary_phi_p + offset + patchsize, &patchPhi[0], patchsize * sizeof(double));
        offset += patchsize * 2;
    } else {
        memcpy(h_boundary_phi_p + offset, &patchPhi[0], patchsize * sizeof(double));
        offset += patchsize;
    } 
}
pEqn_GPU.preProcess(&phi[0], h_boundary_phi_p);

delete[] h_boundary_phi_p;
#endif

// Thermodynamic density needs to be updated by psi*d(p) after the
// pressure solution
const volScalarField psip0(psi*p);

#ifdef GPUSolver_
    // UEqn.H()
    start1 = std::clock();
    volVectorField UEqn_H
    (
        IOobject
        (
            "H("+U.name()+')',
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedVector(dimensionSet(1,-2,-2,0,0,0,0), Zero),
        extrapolatedCalculatedFvPatchScalarField::typeName
    );
    UEqn_GPU.H(&UEqn_H[0][0]);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    UEqn_H.correctBoundaryConditions();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_H_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);

    // UEqn.A()
    start1 = std::clock();
    volScalarField UEqn_A
    (
        IOobject
        (
            "A("+U.name()+')',
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1,-3,-1,0,0,0,0), Zero),
        extrapolatedCalculatedFvPatchScalarField::typeName
    );
    UEqn_GPU.A(&UEqn_A[0]);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    UEqn_A.correctBoundaryConditions();
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_A_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#endif

#ifdef GPUSolverNew_
// volVectorField UEqn_H
// (
//     IOobject
//     (
//         "H("+U.name()+')',
//         runTime.timeName(),
//         mesh,
//         IOobject::NO_READ,
//         IOobject::NO_WRITE
//     ),
//     mesh,
//     dimensionedVector(dimensionSet(1,-2,-2,0,0,0,0), Zero),
//     extrapolatedCalculatedFvPatchScalarField::typeName
// );
// UEqn_GPU.H(&UEqn_H[0][0]);
// UEqn_H.correctBoundaryConditions();

// UEqn.A only needs to compute once

#endif


start2 = std::clock();
#ifdef GPUSolver_
    volScalarField rAU(1.0/UEqn_A);
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn_H, U, p));
#elif defined GPUSolverNew_
    volScalarField rAU(1/UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    Info << "HbyA before\n" << rAU*UEqn.H() << endl;

    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    Info << "HbyA\n" << HbyA << endl;

    UEqn_GPU.getHbyA();
    pEqn_GPU.process();
    UEqn_GPU.sync();

#if defined DEBUG_
    double *h_boundary_rAU = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_rhorAUf = (double*)calloc(dfDataBase.num_boundary_surfaces, sizeof(double));
    double *h_boundary_HbyA = new double[3 * dfDataBase.num_boundary_surfaces];
    offset = 0;
    forAll(rAU.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchrAU = rAU.boundaryField()[patchi];
        const fvPatchVectorField& patchHbyA = HbyA.boundaryField()[patchi];
        const fvsPatchScalarField& patchrhorAUf = rhorAUf.boundaryField()[patchi];
        int patchSize = patchrAU.size();

        if (patchrAU.type() == "processor") {
            memcpy(h_boundary_rAU + offset, &patchrAU[0], patchSize*sizeof(double));
            scalarField patchrAUInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchrAU).patchInternalField()();
            memcpy(h_boundary_rAU + offset + patchSize, &patchrAUInternal[0], patchSize*sizeof(double));

            memcpy(h_boundary_rhorAUf + offset, &patchrhorAUf[0], patchSize*sizeof(double));

            memcpy(h_boundary_HbyA + offset * 3, &patchHbyA[0][0], patchSize*3*sizeof(double));
            vectorField patchHbyAInternal = 
                    dynamic_cast<const processorFvPatchField<vector>&>(patchHbyA).patchInternalField()();
            memcpy(h_boundary_HbyA + offset * 3 + patchSize * 3, &patchHbyAInternal[0][0], patchSize*3*sizeof(double));

            offset += patchSize * 2;
        } else {
            memcpy(h_boundary_rAU + offset, &patchrAU[0], patchSize*sizeof(double));
            memcpy(h_boundary_rhorAUf + offset, &patchrhorAUf[0], patchSize*sizeof(double));
            memcpy(h_boundary_HbyA + offset * 3, &patchHbyA[0][0], patchSize*3*sizeof(double));
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        UEqn_GPU.compareHbyA(&HbyA[0][0], h_boundary_HbyA, false);
        UEqn_GPU.comparerAU(&rAU[0], h_boundary_rAU, false);
        pEqn_GPU.comparerhorAUf(&rhorAUf[0], h_boundary_rhorAUf, false);
    }

    delete h_boundary_rAU;
    delete h_boundary_rhorAUf;
    delete h_boundary_HbyA;
#endif

#else
    volScalarField rAU(1.0/UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    if (pimple.nCorrPiso() <= 1)
    {
        tUEqn.clear();
    }
#endif

surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::interpolate(rho)*fvc::flux(HbyA)
  + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
);

#if defined DEBUG_ && defined GPUSolverNew_
    double *h_boundary_phiHbyA = (double*)calloc(dfDataBase.num_boundary_surfaces, sizeof(double));
    offset = 0;
    forAll(phiHbyA.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchphiHbyA = phiHbyA.boundaryField()[patchi];
        int patchSize = patchphiHbyA.size();
        if (patchphiHbyA.type() == "processor") {
            memcpy(h_boundary_phiHbyA + offset, &patchphiHbyA[0], patchSize*sizeof(double));
            offset += 2 * patchSize;
        } else {
            memcpy(h_boundary_phiHbyA + offset, &patchphiHbyA[0], patchSize*sizeof(double));
            offset += patchSize;
        }
    }
    if (!mpi_init_flag || rank == 0) {
        pEqn_GPU.comparephiHbyA(&phiHbyA[0], h_boundary_phiHbyA, false);
    }
#endif

fvc::makeRelative(phiHbyA, rho, U);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf);

if (pimple.transonic())
{
    surfaceScalarField phid
    (
        "phid",
        (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA
    );

    phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho);

    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA) + fvm::div(phid, p)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();

        start1 = std::clock();
        pEqn.solve();
        end1 = std::clock();
        time_monitor_pEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}
else
{
    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + 
        // psi*correction(fvm::ddt(p)) // TODO: bugs in correction fvMatrix
        psi*fvm::ddt(p)
      + fvc::div(phiHbyA)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));

        start1 = std::clock();
        pEqn.solve();

        #if defined DEBUG_ && defined GPUSolverNew_
        double *h_p = dfDataBase.getFieldPointer("p", location::cpu, position::internal);
        double *h_boundary_p = dfDataBase.getFieldPointer("p", location::cpu, position::boundary);
        offset = 0;
        forAll(p.boundaryField(), patchi)
        {
            const fvPatchScalarField& patchP = p.boundaryField()[patchi];
            int patchsize = patchP.size();
            if (patchP.type() == "processor") {
                memcpy(h_boundary_p + offset, &patchP[0], patchsize * sizeof(double));
                scalarField patchPInternal = 
                        dynamic_cast<const processorFvPatchField<scalar>&>(patchP).patchInternalField()();
                memcpy(h_boundary_p + offset + patchsize, &patchPInternal[0], patchsize * sizeof(double));
                offset += patchsize * 2;
            } else {
                memcpy(h_boundary_p + offset, &patchP[0], patchsize * sizeof(double));
                offset += patchsize;
            }
        }
        memcpy(h_p, &p[0], dfDataBase.cell_value_bytes);
        pEqn_GPU.correctP(h_p, h_boundary_p);
        // if (!mpi_init_flag || rank == 0) {
        //     pEqn_GPU.comparep(h_p, h_boundary_p, false);
        // }
        #endif
        
        #ifdef GPUSolverNew_
        pEqn_GPU.postProcess();
        #endif

        end1 = std::clock();
        time_monitor_pEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }

        #if defined DEBUG_ && defined GPUSolverNew_
        std::vector<double> h_internal_coeffs(dfDataBase.num_boundary_surfaces);
        std::vector<double> h_boundary_coeffs(dfDataBase.num_boundary_surfaces);

        offset = 0;
        for (int patchi = 0; patchi < dfDataBase.num_patches; patchi++)
        {
            const fvPatchScalarField& patchP = p.boundaryField()[patchi];
            int patchsize = dfDataBase.patch_size[patchi];
            const double* internal_coeff_ptr = &pEqn.internalCoeffs()[patchi][0];
            const double* boundary_coeff_ptr = &pEqn.boundaryCoeffs()[patchi][0];
            memcpy(h_internal_coeffs.data() + offset, internal_coeff_ptr, patchsize * sizeof(double));
            memcpy(h_boundary_coeffs.data() + offset, boundary_coeff_ptr, patchsize * sizeof(double));
            if (patchP.type() == "processor") offset += 2 * patchsize;
            else offset += patchsize;
        }
        // if (!mpi_init_flag || rank == 0) {
        //     pEqn_GPU.compareResult(&pEqn.lower()[0], &pEqn.upper()[0], &pEqn.diag()[0], &pEqn.source()[0], 
        //             h_internal_coeffs.data(), h_boundary_coeffs.data(), false);
        // }

        // surfaceScalarField testFlux("testFlux", pEqn.flux());
        // double *h_boundary_flux = new double[dfDataBase.num_boundary_surfaces];
        // offset = 0;
        // forAll(testFlux.boundaryField(), patchi)
        // {
        //     const scalarField& patchFlux = testFlux.boundaryField()[patchi];
        //     int patchSize = patchFlux.size();
        //     memcpy(h_boundary_flux + offset, &patchFlux[0], patchSize*sizeof(double));
        //     offset += patchSize;
        // }
        // pEqn_GPU.comparephiFlux(&testFlux[0], h_boundary_flux, false);
        
        double *h_boundary_phi = new double[dfDataBase.num_boundary_surfaces];
        offset = 0;
        forAll(phi.boundaryField(), patchi)
        {
            const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
            int patchSize = patchFlux.size();
            memcpy(h_boundary_phi + offset, &patchFlux[0], patchSize*sizeof(double));
            if (patchFlux.type() == "processor") {
                memset(h_boundary_phi + offset + patchSize, 0, patchSize*sizeof(double));
                offset += 2 * patchSize;
            } else {
                offset += patchSize;
            }
        }
        if (!mpi_init_flag || rank == 0) {
            pEqn_GPU.comparephi(&phi[0], h_boundary_phi, false);
        }
        #endif
    }
}
 
bool limitedp = pressureControl.limit(p);

// Thermodynamic density update
thermo.correctRho(psi*p - psip0);

// limitedp = false
if (limitedp)
{
    rho = thermo.rho();
}

//#undef CPUSolver_
//#define GPUSolverNew_

double *h_rho_p = dfDataBase.getFieldPointer("rho", location::cpu, position::internal);
double *h_boundary_rho_p = dfDataBase.getFieldPointer("rho", location::cpu, position::boundary);
offset = 0;
forAll(rho.boundaryField(), patchi)
{
    const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
    int patchsize = patchRho.size();
    if (patchRho.type() == "processor") {
        memcpy(h_boundary_rho_p + offset, &patchRho[0], patchsize * sizeof(double));
        scalarField patchRhoInternal = 
                dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).patchInternalField()();
        memcpy(h_boundary_rho_p + offset + patchsize, &patchRhoInternal[0], patchsize * sizeof(double));
        offset += patchsize * 2;
    } else {
        memcpy(h_boundary_rho_p + offset, &patchRho[0], patchsize * sizeof(double));
        offset += patchsize;
    }
}
memcpy(h_rho_p, &rho[0], dfDataBase.cell_value_bytes);
rhoEqn_GPU.correctPsi(h_rho_p, h_boundary_rho_p);


double *h_phi_p = new double[dfDataBase.num_surfaces];
double *h_boundary_phi_p = new double[dfDataBase.num_boundary_surfaces];
offset = 0;
forAll(phi.boundaryField(), patchi)
{
    const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
    int patchsize = patchPhi.size();

    if (patchPhi.type() == "processor") {
        memcpy(h_boundary_phi_p + offset, &patchPhi[0], patchsize * sizeof(double));
        memcpy(h_boundary_phi_p + offset + patchsize, &patchPhi[0], patchsize * sizeof(double));
        offset += patchsize * 2;
    } else {
        memcpy(h_boundary_phi_p + offset, &patchPhi[0], patchsize * sizeof(double));
        offset += patchsize;
    } 
}
memcpy(h_phi_p, &phi[0], dfDataBase.surface_value_bytes);
pEqn_GPU.preProcess(&phi[0], h_boundary_phi_p);
delete []h_phi_p;
delete []h_boundary_phi_p;

#include "rhoEqn.H"
#include "compressibleContinuityErrs.H" // TODO: implement this func in future

//#undef GPUSolverNew_
//#define CPUSolver_
// Explicitly relax pressure for momentum corrector
// p.relax();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);

#if defined DEBUG_ && defined GPUSolverNew_
// check U
double *h_boundary_u_tmp = new double[dfDataBase.num_boundary_surfaces * 3];
offset = 0;
forAll(U.boundaryField(), patchi)
{
    const fvPatchVectorField& patchU = U.boundaryField()[patchi];
    int patchSize = patchU.size();

    if (patchU.type() == "processor") {
        memcpy(h_boundary_u_tmp + 3*offset, &patchU[0][0], patchSize*sizeof(double)*3);
        vectorField patchUInternal = 
                dynamic_cast<const processorFvPatchField<vector>&>(patchU).patchInternalField()();
        memcpy(h_boundary_u_tmp + 3*offset + 3*patchSize, &patchUInternal[0][0], patchSize*sizeof(double)*3);
        offset += patchSize * 2;
    } else {
        memcpy(h_boundary_u_tmp + offset, &patchU[0][0], patchSize*sizeof(double)*3);
        offset += patchSize;
    }
}
if (!mpi_init_flag || rank == 0) {
    pEqn_GPU.compareU(&U[0][0], h_boundary_u_tmp, false);
}
#endif

#ifdef GPUSolver_
    start1 = std::clock();
    UEqn_GPU.correctPsi(&U[0][0]);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#elif defined GPUSolverNew_
    UEqn_GPU.correctPsi(&U[0][0], h_boundary_u_tmp);
    delete h_boundary_u_tmp;
#endif

if (pimple.simpleRho())
{
    rho = thermo.rho();
}

// Correct rhoUf if the mesh is moving
fvc::correctRhoUf(rhoUf, rho, U, phi);
DEBUG_TRACE;
if (thermo.dpdt())
{
    dpdt = fvc::ddt(p);

    if (mesh.moving())
    {
        dpdt -= fvc::div(fvc::meshPhi(rho, U), p);
    }
}

#if defined DEBUG_ && defined GPUSolverNew_
// pEqn_GPU.comparedpdt(&dpdt[0], false);
#endif


end2 = std::clock();
time_monitor_pEqn += double(end2 - start2) / double(CLOCKS_PER_SEC);

// #undef CPUSolver_
// #define GPUSolverNew_
