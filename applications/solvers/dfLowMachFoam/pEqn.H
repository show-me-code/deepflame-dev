if (!pimple.simpleRho())
{
    rho = thermo.rho();
}

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
UEqn_H.correctBoundaryConditions();

// UEqn.A only needs to compute once

#endif


start2 = std::clock();
#ifdef GPUSolver_
    volScalarField rAU(1.0/UEqn_A);
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn_H, U, p));
#elif defined GPUSolverNew_
    volScalarField rAU(1/UEqn_A);
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    volVectorField HbyA(constrainHbyA(rAU*UEqn_H, U, p));

    UEqn_GPU.getHbyA();
    UEqn_GPU.sync();

#if defined DEBUG_
    double *h_boundary_rAU = new double[dfDataBase.num_boundary_surfaces];
    double *h_boundary_HbyA = new double[3 * dfDataBase.num_boundary_surfaces];
    offset = 0;
    forAll(rAU.boundaryField(), patchi)
    {
        const scalarField& patchrAU = rAU.boundaryField()[patchi];
        const vectorField& patchHbyA = HbyA.boundaryField()[patchi];
        int patchSize = patchrAU.size();
        memcpy(h_boundary_rAU + offset, &patchrAU[0], patchSize*sizeof(double));
        memcpy(h_boundary_HbyA + offset * 3, &patchHbyA[0][0], patchSize*3*sizeof(double));
        offset += patchSize;
    }
    UEqn_GPU.compareHbyA(&HbyA[0][0], h_boundary_HbyA, false);
    UEqn_GPU.comparerAU(&rAU[0], h_boundary_rAU, false);
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
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAUf, p));

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

bool limitedp = pressureControl.limit(p);

// Thermodynamic density update
thermo.correctRho(psi*p - psip0);

if (limitedp)
{
    rho = thermo.rho();
}

#include "rhoEqn.H"
#include "compressibleContinuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
K = 0.5*magSqr(U);

#ifdef GPUSolver_
    start1 = std::clock();
    UEqn_GPU.correctPsi(&U[0][0]);
    end1 = std::clock();
    time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_UEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
#elif defined GPUSolverNew_
    UEqn_GPU.correctPsi(&U[0][0]);
#endif

if (pimple.simpleRho())
{
    rho = thermo.rho();
}

// Correct rhoUf if the mesh is moving
fvc::correctRhoUf(rhoUf, rho, U, phi);

if (thermo.dpdt())
{
    dpdt = fvc::ddt(p);

    if (mesh.moving())
    {
        dpdt -= fvc::div(fvc::meshPhi(rho, U), p);
    }
}
end2 = std::clock();
time_monitor_pEqn += double(end2 - start2) / double(CLOCKS_PER_SEC);

