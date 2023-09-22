/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Global
    rhoEqn

Description
    Solve the continuity for density.

\*---------------------------------------------------------------------------*/
//#undef GPUSolverNew_
//#define CPUSolver_
#ifdef GPUSolver_
{
    start1 = std::clock();
    rho.oldTime();

    int offset = 0;
    forAll(U.boundaryField(), patchi)
    {
        const fvsPatchScalarField& patchFlux = phi.boundaryField()[patchi];
        int patchSize = patchFlux.size();
        memcpy(boundary_phi_init+offset, &patchFlux[0], patchSize*sizeof(double));
        offset += patchSize;
    }
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_mtxAssembly_CPU_prepare += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    rhoEqn_GPU.initializeTimeStep();
    rhoEqn_GPU.fvc_div(&phi[0], boundary_phi_init);
    rhoEqn_GPU.fvm_ddt(&rho.oldTime()[0]);
    rhoEqn_GPU.sync();
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_mtxAssembly_GPU_run += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    rhoEqn_GPU.updatePsi(&rho.primitiveFieldRef()[0]);
    rho.correctBoundaryConditions();
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_correctBC += double(end1 - start1) / double(CLOCKS_PER_SEC);
}
#elif defined GPUSolverNew_
    rho.oldTime();

    double *h_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::internal);
    double *h_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::internal);
    double *h_boundary_phi = dfDataBase.getFieldPointer("phi", location::cpu, position::boundary);
    double *h_boundary_rho = dfDataBase.getFieldPointer("rho", location::cpu, position::boundary);
    memcpy(h_phi, &phi[0], dfDataBase.surface_value_bytes);

    int offset = 0;
    forAll(rho.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
        int patchsize = patchRho.size();
        if (patchRho.type() == "processor") {
            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            scalarField patchRhoInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).patchInternalField()();
            memcpy(h_boundary_rho + offset + patchsize, &patchRhoInternal[0], patchsize * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            offset += patchsize;
        }
    }
    offset = 0;
    forAll(phi.boundaryField(), patchi) 
    {
        const fvsPatchScalarField& patchPhi = phi.boundaryField()[patchi];
        int patchsize = patchPhi.size();
        memcpy(h_boundary_phi + offset, &patchPhi[0], patchsize * sizeof(double));
        offset += patchsize;
    }

    rhoEqn_GPU.preProcess(h_phi, h_boundary_phi, h_boundary_rho);

    rhoEqn_GPU.process();

    rhoEqn_GPU.postProcess(h_rho);
    memcpy(&rho[0], h_rho, dfDataBase.cell_value_bytes);

    rho.correctBoundaryConditions();

    // check boundary rho
    offset = 0;
    forAll(rho.boundaryField(), patchi)
    {
        const fvPatchScalarField& patchRho = rho.boundaryField()[patchi];
        int patchsize = patchRho.size();
        if (patchRho.type() == "processor") {
            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            scalarField patchRhoInternal = 
                    dynamic_cast<const processorFvPatchField<scalar>&>(patchRho).patchInternalField()();
            memcpy(h_boundary_rho + offset + patchsize, &patchRhoInternal[0], patchsize * sizeof(double));
            offset += patchsize * 2;
        } else {
            memcpy(h_boundary_rho + offset, &patchRho[0], patchsize * sizeof(double));
            offset += patchsize;
        }
    }
    int rank = -1;
    if (mpi_init_flag) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (!mpi_init_flag || rank == 0) {
        rhoEqn_GPU.compareRho(&rho[0], h_boundary_rho, false);
    }

#ifdef DEBUG_
    // checkValue
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );
    rhoEqn.solve();

    if (!mpi_init_flag || rank == 0) {
        rhoEqn_GPU.compareResult(&rhoEqn.diag()[0], &rhoEqn.source()[0], false);
    }
#endif
#else
{
    start1 = std::clock();
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
    );
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    rhoEqn.solve();
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
}
#endif
//#undef CPUSolver_
//#define GPUSolverNew_
// ************************************************************************* //
