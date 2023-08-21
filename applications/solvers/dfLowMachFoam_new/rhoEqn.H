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

// ************************************************************************* //
