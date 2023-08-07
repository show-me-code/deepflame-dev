/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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

Application
    unittest

Description
    GPU unittest

\*---------------------------------------------------------------------------*/

#include "dfChemistryModel.H"
#include "CanteraMixture.H"
// #include "hePsiThermo.H"
#include "heRhoThermo.H"

#include "fvCFD.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "PstreamGlobals.H"
#include "basicThermo.H"
#include "CombustionModel.H"

#include <cuda_runtime.h>
#include <thread>
#include "upwind.H"

#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"
#include "createGPUSolver.H"
#include "GPUTestBase.H"

int main(int argc, char *argv[])
{
#ifdef USE_PYTORCH
    pybind11::scoped_interpreter guard{};//start python interpreter
#endif
    #include "postProcess.H"

    // #include "setRootCaseLists.H"
    #include "listOptions.H"
    #include "setRootCase2.H"
    #include "listOutput.H"

    #include "createTime.H"
    #include "createMesh.H"
    #include "createDyMControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createRhoUfIfPresent.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"
        }

        createGPUBase(mesh, Y);

        // unittest of fvm::ddt(rho, U)
        test_fvm_ddt_vector(dfDataBase, mesh, rho, U, initType::original);
        test_fvm_ddt_vector(dfDataBase, mesh, rho, U, initType::randomInit);
        // unittest of fvm::div(phi, U)
        test_fvm_div_vector(dfDataBase, mesh, phi, U, initType::original);
        test_fvm_div_vector(dfDataBase, mesh, phi, U, initType::randomInit);
        // unittest of fvm::laplacian(gamma, U)
        const tmp<volScalarField> nuEff_tmp(turbulence->nuEff());
        const volScalarField& nuEff = nuEff_tmp();
        volScalarField gamma = rho * nuEff;
        test_fvm_laplacian_vector(dfDataBase, mesh, gamma, U, initType::original);
        test_fvm_laplacian_vector(dfDataBase, mesh, gamma, U, initType::randomInit);
    }
    return 0;
}

