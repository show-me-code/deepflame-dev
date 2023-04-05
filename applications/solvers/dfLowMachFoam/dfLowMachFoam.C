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
    rhoPimpleFoam

Description
    Transient solver for turbulent flow of compressible fluids for HVAC and
    similar applications, with optional mesh motion and mesh topology changes.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient simulations.

\*---------------------------------------------------------------------------*/

#include "dfChemistryModel.H"
#include "CanteraMixture.H"
#include "hePsiThermo.H"

#ifdef USE_PYTORCH
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> //used to convert
#endif

#ifdef USE_LIBTORCH
#include <torch/script.h>
#include "DNNInferencer.H"
#endif

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

#include "dfMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

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

    double time_monitor_flow=0;
    double time_monitor_chem=0;
    double time_monitor_Y=0;
    double time_monitor_E=0;
    double time_monitor_corrThermo=0;
    double time_monitor_corrDiff=0;
    label timeIndex = 0;
    clock_t start, end;

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    #include "createdfSolver.H"

//     double time_permutate_Field_Variable_CPU=0;
// double time_permutate_Field_Variable_GPU=0;
// double time_construct_Mesh_Variable=0;
// double time_construct_Field_Variable=0;
// start = std::clock(); // start construct Mesh Variable
// // extract mesh variable from OpenFoam
// const labelUList& owner = mesh.owner();
// const labelUList& neighbour = mesh.neighbour();
// int num_cells = mesh.nCells();
// int num_surfaces = neighbour.size();
// double rdelta_t = 1/1e-6;

// // initialize variables
// // - the pre-permutated and post-permutated interpolation weight list
// std::vector<double> h_weight_vec_init(2*num_surfaces),h_weight_vec(2*num_surfaces);
// // - the pre-permutated and post-permutated flux (phi) list
// std::vector<double> h_phi_vec_init(2*num_surfaces),h_phi_vec(2*num_surfaces);
// // - the pre-permutated and post-permutated cell face vector list
// std::vector<double> h_surface_vector_vec_init(2*num_surfaces*3),h_surface_vector_vec(2*num_surfaces*3);
// // - the size of off-diagnal entries
// int offDiag_bytes = 2*num_surfaces*sizeof(double);
// // - the device pointer to rho_new, rho_old, velocity_old, pressure and volume list
// double *d_rho_new = nullptr, *d_rho_old = nullptr, *d_velocity_old = nullptr, 
// *d_pressure = nullptr, *d_volume = nullptr;
// // - the device pointer to the pre-permutated and post-permutated interpolation weight list
// double *d_weight_init = nullptr, *d_weight = nullptr;
// // - the device pointer to the pre-permutated and post-permutated flux (phi) list
// double *d_phi_init = nullptr, *d_phi = nullptr;
// // - the device pointer to the pre-permutated and post-permutated cell face vector list
// double *d_surface_vector_init = nullptr, *d_surface_vector = nullptr;
// // - the device pointer to the permutated index list
// int *d_permedIndex = NULL;
// // - allocate memory on device
// cudaMalloc((void **)&d_weight_init, offDiag_bytes);
// cudaMalloc((void **)&d_surface_vector_init, 3*offDiag_bytes);
// cudaMalloc((void **)&d_phi_init, offDiag_bytes);
// cudaMalloc((void **)&d_weight, offDiag_bytes);
// cudaMalloc((void **)&d_surface_vector, 3*offDiag_bytes);
// cudaMalloc((void **)&d_phi, offDiag_bytes);
// cudaMalloc((void **)&d_permedIndex, offDiag_bytes);

// /************************construct mesh variables****************************/
// // - h_csr_row_index & h_csr_diag_index
// std::vector<int> h_mtxEntry_perRow_vec(num_cells);
// std::vector<int> h_csr_diag_index_vec(num_cells);
// std::vector<int> h_csr_row_index_vec(num_cells + 1, 0);

// for (int faceI = 0; faceI < num_surfaces; faceI++)
// {
//     h_csr_diag_index_vec[neighbour[faceI]]++;
//     h_mtxEntry_perRow_vec[neighbour[faceI]]++;
//     h_mtxEntry_perRow_vec[owner[faceI]]++;
// }
// // - consider diagnal element in each row
// std::transform(h_mtxEntry_perRow_vec.begin(), h_mtxEntry_perRow_vec.end(), h_mtxEntry_perRow_vec.begin(), [](int n)
//     {return n + 1;});
// // - construct h_csr_row_index & h_csr_diag_index
// std::partial_sum(h_mtxEntry_perRow_vec.begin(), h_mtxEntry_perRow_vec.end(), h_csr_row_index_vec.begin()+1);
// int *h_csr_row_index = h_csr_row_index_vec.data();
// int *h_csr_diag_index = h_csr_diag_index_vec.data();

// // - h_csr_col_index
// std::vector<int> rowIndex(2*num_surfaces + num_cells), colIndex(2*num_surfaces + num_cells), diagIndex(num_cells);
// std::iota(diagIndex.begin(), diagIndex.end(), 0);
// // initialize the RowIndex (rowIndex of lower + upper + diagnal)
// std::copy(&neighbour[0], &neighbour[0] + num_surfaces, rowIndex.begin());
// std::copy(&owner[0], &owner[0] + num_surfaces, rowIndex.begin() + num_surfaces);
// std::copy(diagIndex.begin(), diagIndex.end(), rowIndex.begin() + 2*num_surfaces);
// // initialize the ColIndex (colIndex of lower + upper + diagnal)
// std::copy(&owner[0], &owner[0] + num_surfaces, colIndex.begin());
// std::copy(&neighbour[0], &neighbour[0] + num_surfaces, colIndex.begin() + num_surfaces);
// std::copy(diagIndex.begin(), diagIndex.end(), colIndex.begin() + 2*num_surfaces);

// // - construct hashTable for sorting
// std::multimap<int,int> rowColPair;
// for (int i = 0; i < 2*num_surfaces+num_cells; i++)
// {
//     rowColPair.insert(std::make_pair(rowIndex[i], colIndex[i]));
// }
// // - sort
// std::vector<std::pair<int, int>> globalPerm(rowColPair.begin(), rowColPair.end());
// std::sort(globalPerm.begin(), globalPerm.end(), []
// (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
// if (pair1.first != pair2.first) {
//     return pair1.first < pair2.first;
// } else {
//     return pair1.second < pair2.second;
// }
// });

// std::vector<int> h_csr_col_index_vec;
// std::transform(globalPerm.begin(), globalPerm.end(), std::back_inserter(h_csr_col_index_vec), []
//     (const std::pair<int, int>& pair) {
//     return pair.second;
// });
// int *h_csr_col_index = h_csr_col_index_vec.data();

// /************************construct permutation list****************************/
// std::vector<int> offdiagRowIndex(2*num_surfaces), permIndex(2*num_surfaces);
// // - initialize the offdiagRowIndex (rowIndex of lower + rowIndex of upper)
// std::copy(&neighbour[0], &neighbour[0] + num_surfaces, offdiagRowIndex.begin());
// std::copy(&owner[0], &owner[0] + num_surfaces, offdiagRowIndex.begin() + num_surfaces);

// // - initialize the permIndex (0, 1, ..., 2*num_surfaces)
// std::iota(permIndex.begin(), permIndex.end(), 0);

// // - construct hashTable for sorting
// std::multimap<int,int> permutation;
// for (int i = 0; i < 2*num_surfaces; i++)
// {
//     permutation.insert(std::make_pair(offdiagRowIndex[i], permIndex[i]));
// }
// // - sort 
// std::vector<std::pair<int, int>> permPair(permutation.begin(), permutation.end());
// std::sort(permPair.begin(), permPair.end(), []
// (const std::pair<int, int>& pair1, const std::pair<int, int>& pair2){
//     if (pair1.first != pair2.first) {
//         return pair1.first < pair2.first;
//     } else {
//         return pair1.second < pair2.second;
//     }
// });
// // - form permedIndex list
// std::vector<int> permedIndex;
// std::transform(permPair.begin(), permPair.end(), std::back_inserter(permedIndex), []
//     (const std::pair<int, int>& pair) {
//     return pair.second;
// });
// end = std::clock();// end construct Mesh Variable
// time_construct_Mesh_Variable += double(end - start) / double(CLOCKS_PER_SEC);

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        timeIndex ++;

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

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (splitting)
            {
                #include "YEqn_RR.H"
            }
            if (pimple.firstPimpleIter() || moveMeshOuterCorrectors)
            {
                // Store momentum to set rhoUf for introduced faces.
                autoPtr<volVectorField> rhoU;
                if (rhoUf.valid())
                {
                    rhoU = new volVectorField("rhoU", rho*U);
                }
            }

            if (pimple.firstPimpleIter() && !pimple.simpleRho())
            {
                #include "rhoEqn.H"
            }

            start = std::clock();
            #include "UEqn.H"
            end = std::clock();
            time_monitor_flow += double(end - start) / double(CLOCKS_PER_SEC);

            if(combModelName!="ESF" && combModelName!="flareFGM" )
            {
                #include "YEqn.H"

                start = std::clock();
                #include "EEqn.H"
                end = std::clock();
                time_monitor_E += double(end - start) / double(CLOCKS_PER_SEC);

                start = std::clock();
                chemistry->correctThermo();
                end = std::clock();
                time_monitor_corrThermo += double(end - start) / double(CLOCKS_PER_SEC);
            }
            else
            {
                combustion->correct();
            }

            Info<< "min/max(T) = " << min(T).value() << ", " << max(T).value() << endl;

            // --- Pressure corrector loop

            start = std::clock();
            while (pimple.correct())
            {
                if (pimple.consistent())
                {
                    #include "pcEqn.H"
                }
                else
                {
                    #include "pEqn.H"
                }
            }
            end = std::clock();
            time_monitor_flow += double(end - start) / double(CLOCKS_PER_SEC);

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        rho = thermo.rho();

        runTime.write();

        Info << "output time index " << runTime.timeIndex() << endl;

        Info<< "========Time Spent in diffenet parts========"<< endl;
        Info<< "Chemical sources           = " << time_monitor_chem << " s" << endl;
        Info<< "Species Equations          = " << time_monitor_Y << " s" << endl;
        Info<< "U & p Equations            = " << time_monitor_flow << " s" << endl;
        Info<< "Energy Equations           = " << time_monitor_E << " s" << endl;
        Info<< "thermo & Trans Properties  = " << time_monitor_corrThermo << " s" << endl;
        Info<< "Diffusion Correction Time  = " << time_monitor_corrDiff << " s" << endl;
        Info<< "sum Time                   = " << (time_monitor_chem + time_monitor_Y + time_monitor_flow + time_monitor_E + time_monitor_corrThermo + time_monitor_corrDiff) << " s" << endl;
        Info<< "============================================"<<nl<< endl;

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s" << endl;
#ifdef USE_PYTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry->time_allsolve() << " s"
            << "    submasterTime = " << chemistry->time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry->time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry->time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry->time_sendRecvSolution() << " s" << nl
            << "    getDNNinputsTime = " << chemistry->time_getDNNinputs() << " s"
            << "    DNNinferenceTime = " << chemistry->time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << nl
            << "    vec2ndarrayTime = " << chemistry->time_vec2ndarray() << " s"
            << "    pythonTime = " << chemistry->time_python() << " s"<< nl << endl;
        }
#endif
#ifdef USE_LIBTORCH
        if (log_ && torch_)
        {
            Info<< "    allsolveTime = " << chemistry->time_allsolve() << " s"
            << "    submasterTime = " << chemistry->time_submaster() << " s" << nl
            << "    sendProblemTime = " << chemistry->time_sendProblem() << " s"
            << "    recvProblemTime = " << chemistry->time_RecvProblem() << " s"
            << "    sendRecvSolutionTime = " << chemistry->time_sendRecvSolution() << " s" << nl
            << "    DNNinferenceTime = " << chemistry->time_DNNinference() << " s"
            << "    updateSolutionBufferTime = " << chemistry->time_updateSolutionBuffer() << " s" << nl;
        }
#endif
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
