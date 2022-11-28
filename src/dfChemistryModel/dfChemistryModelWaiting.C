/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
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

\*---------------------------------------------------------------------------*/

#include "dfChemistryModel.H"
#include "UniformField.H"
#include "clockTime.H"
#include "runtime_assert.H"
#include <unistd.h>
#include <omp.h>




// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::dfChemistryModel<ThermoType>::dfChemistryModel
(
    ThermoType& thermo
)
:
    IOdictionary
    (
        IOobject
        (
            "CanteraTorchProperties",
            thermo.db().time().constant(),
            thermo.db(),
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),
    thermo_(thermo),
    mixture_(dynamic_cast<CanteraMixture&>(thermo)),
    CanteraGas_(mixture_.CanteraGas()),
    mesh_(thermo.p().mesh()),
    chemistry_(lookup("chemistry")),
    relTol_(this->subDict("odeCoeffs").lookupOrDefault("relTol",1e-9)),
    absTol_(this->subDict("odeCoeffs").lookupOrDefault("absTol",1e-15)),
    Y_(mixture_.Y()),
    rhoD_(mixture_.nSpecies()),
    hai_(mixture_.nSpecies()),
    hc_(mixture_.nSpecies()),
    yTemp_(mixture_.nSpecies()),
    dTemp_(mixture_.nSpecies()),
    hrtTemp_(mixture_.nSpecies()),
    cTemp_(mixture_.nSpecies()),
    RR_(mixture_.nSpecies()),
    alpha_(const_cast<volScalarField&>(thermo.alpha())),
    T_(thermo.T()),
    p_(thermo.p()),
    rho_(mesh_.objectRegistry::lookupObject<volScalarField>("rho")),
    mu_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).mu()())),
    psi_(const_cast<volScalarField&>(dynamic_cast<psiThermo&>(thermo).psi())),
    time_allsolve_(0),
    time_submaster_(0),
    time_sendProblem_(0),
    time_RecvProblem_(0),
    time_sendRecvSolution_(0),
    time_getDNNinputs_(0),
    time_DNNinference_(0),
    time_updateSolutionBuffer_(0),
    time_vec2ndarray_(0),
    time_python_(0),
    Qdot_
    (
        IOobject
        (
            "Qdot",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
    ),
    torchSwitch_(lookupOrDefault("torch", false)),
    balancer_(createBalancer()),
    cpuTimes_
    (
        IOobject
        (
            "cellCpuTimes",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh_,
        scalar(0.0)
    )
{
    if(torchSwitch_)
    {
        torchModelName_ = this->lookupOrDefault("torchModel", word(""));

        Tact1_ = this->subDict("torchParameters1").lookupOrDefault("Tact", 700);
        Qdotact1_ = this->subDict("torchParameters1").lookupOrDefault("Qdotact", 1e9);

        Tact2_ = this->subDict("torchParameters2").lookupOrDefault("Tact", 700);
        Qdotact2_ = this->subDict("torchParameters2").lookupOrDefault("Qdotact", 1e9);

        Tact3_ = this->subDict("torchParameters3").lookupOrDefault("Tact", 700);
        Qdotact3_ = this->subDict("torchParameters3").lookupOrDefault("Qdotact", 1e9);
        
        coresPerGPU = this->subDict("torchParameters1").lookupOrDefault("coresPerGPU", 8);
    }

    for(const auto& name : CanteraGas_->speciesNames())
    {
        species_.append(name);
    }
    forAll(RR_, fieldi)
    {
        RR_.set
        (
            fieldi,
            new volScalarField::Internal
            (
                IOobject
                (
                    "RR." + Y_[fieldi].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimMass/dimVolume/dimTime, 0)
            )
        );
    }

    forAll(rhoD_, i)
    {
        rhoD_.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                    "rhoD_" + Y_[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimDensity*dimViscosity, 0) // kg/m/s
            )
        );
    }

    forAll(hai_, i)
    {
        hai_.set
        (
            i,
            new volScalarField
            (
                IOobject
                (
                    "hai_" + Y_[i].name(),
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimEnergy/dimMass, 0)
            )
        );
    }
    if(balancer_.log())
    {
        cpuSolveFile_ = logFile("cpu_solve.out");
        cpuSolveFile_() << "                  time" << tab
                        << "           getProblems" << tab  
                        << "           updateState" << tab
                        << "               balance" << tab
                        << "           solveBuffer" << tab
                        << "             unbalance" << tab
                        << "               rank ID" << endl;
    }

    Info<<"--- I am here in Cantera-construct ---"<<endl;
    Info<<"relTol_ === "<<relTol_<<endl;
    Info<<"absTol_ === "<<absTol_<<endl;

    forAll(hc_, i)
    {
        hc_[i] = CanteraGas_->Hf298SS(i)/CanteraGas_->molecularWeight(i);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &myWorldComm);
    localMpiCommSplit(8); 
    Info << "sub communcator construct complete!" << endl;
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::dfChemistryModel<ThermoType>::
~dfChemistryModel()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template <class ThermoType>
void Foam::dfChemistryModel<ThermoType>::localMpiCommSplit(int cpuPerNode)
{
    //Info << "mpi total ranks " << numranks << endl;
    int world_rank, world_size;
    MPI_Comm_rank(myWorldComm, &world_rank);
    MPI_Comm_size(myWorldComm, &world_size);
    //Info << "size and rank " << world_rank << world_size << endl;

    int color = world_rank/cpuPerNode;
    MPI_Comm local;
    MPI_Comm_split(myWorldComm, color, world_rank, &local); // divide comm from two colors

    localComm = local;
    int rank, size;
    MPI_Comm_rank(localComm, &rank);
    MPI_Comm_size(localComm, &size);
    std::cout << "local size and rank " << rank << size << " \n"<< std::endl;
}

template<class ThermoType>
template<class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const DeltaTType& deltaT
)
{
    scalar result = 0;
    if(torchSwitch_)
    {
        result = torchDCUSolve(deltaT);
    }
    else
    {
        result = canteraSolve(deltaT);
    }
    return result;
}

template<class ThermoType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const scalar deltaT
)
{
    // Don't allow the time-step to change more than a factor of 2
    return min
    (
        this->solve<UniformField<scalar>>(UniformField<scalar>(deltaT)),
        2*deltaT
    );
}


template<class ThermoType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve
(
    const scalarField& deltaT
)
{
    return this->solve<scalarField>(deltaT);
}


template<class ThermoType>
template<class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::canteraSolve
(
    const DeltaTType& deltaT
)
{
    scalar deltaTMin = great;

    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    Info<<"=== begin Cantera-solve === "<<endl;

    Cantera::Reactor react;
    //Cantera::IdealGasReactor react;  // Constant-UV, default, constant Volumn
    //Cantera::IdealGasConstPressureReactor react;  // Constant-HP, constant pressure

    scalarField c0(CanteraGas_->nSpecies());

    Qdot_ = Zero;

    Info << "Y_Chem" << Y_[0][826]<<endl;

    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];
        try
        {
            for (size_t i=0; i<CanteraGas_->nSpecies(); ++i)
            {
                yTemp_[i] = Y_[i][cellI];
            }

            CanteraGas_->setState_TPY(Ti, pi, yTemp_.begin());
            CanteraGas_->getConcentrations(c0.begin()); // value --> c0

            react.insert(mixture_.CanteraSolution());
            react.setEnergy(0); // keep T const before and after sim.advance. this will give you a little improvement
            Cantera::ReactorNet sim;
            sim.addReactor(react);
            setNumerics(sim);


            sim.advance(deltaT[cellI]);


            CanteraGas_->getConcentrations(cTemp_.begin()); // value --> cTemp_

            for (size_t i=0; i<CanteraGas_->nSpecies(); ++i)
            {
                RR_[i][cellI] = (cTemp_[i] - c0[i])*CanteraGas_->molecularWeight(i)/deltaT[cellI];
            }
            // CanteraGas_->molecularWeight(i)    kg/kmol

            forAll(Y_, i)
            {
                Qdot_[cellI] -= hc_[i]*RR_[i][cellI];
            }
        }
        catch(Cantera::CanteraError& err)
        {
            // handle exceptions thrown by Cantera
            std::cout << err.what() << std::endl;

            FatalErrorIn("dfChemistryModel::solve")
                << " Cantera complained in cell " << cellI
                << " with a Cantera::CanteraError"  << endl
                << abort(FatalError) ;
        }
    }

    Info<<"=== end Cantera-solve === "<<endl;
    return deltaTMin;
}


template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::setNumerics(Cantera::ReactorNet &sim)
{
    sim.setTolerances(relTol_,absTol_);
}

template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::torchCUDAoneCoreSolve(
    const DeltaTType &deltaT)
{
    scalar deltaTMin = great;

    if (!this->chemistry_)
    {
        return deltaTMin;
    }
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Info << "=== begin torchCUDAoneCore-solve === " << endl;

    // set variables
    scalarList yPre_(mixture_.nSpecies());
    scalarList yBCT_(mixture_.nSpecies());
    scalarList u_(mixture_.nSpecies() + 2);
    Cantera::Reactor react;
    double lambda = 0.1;

    std::vector<size_t> torch_cell;
    label torch_cellname = 0;

    // obtain the number of DNN cells
    std::chrono::steady_clock::time_point start_0 = std::chrono::steady_clock::now();
    forAll(T_, cellI)
    {
        if (T_[cellI] >= Tact1_)
        {
            torch_cell.push_back(cellI);
        }
    }

    // generate GPU inputs and solve CVODE cells
    std::vector<double> inputs_;
    inputs_.reserve(torch_cell.size() * (CanteraGas_->nSpecies() + 3));

    forAll(T_, cellI)
    {
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];
        scalar rhoi = rho_[cellI];

        if (Ti >= Tact1_)
        {
            Qdot_[cellI] = 0.0;

            // set inputs
            inputs_.push_back(rhoi);
            inputs_.push_back(Ti);
            inputs_.push_back(pi/101325);
            for (size_t i = 0; i < CanteraGas_->nSpecies(); i++)
            {
                inputs_.push_back(Y_[i][cellI]);
            }
        }
        else
        {
            Qdot_[cellI] = 0.0;
            for (size_t i = 0; i < CanteraGas_->nSpecies(); i++)
            {
                yPre_[i] = Y_[i][cellI];
            }

            CanteraGas_->setState_TPY(Ti, pi, yPre_.begin());
            react.insert(mixture_.CanteraSolution());
            react.setEnergy(0);

            Cantera::ReactorNet sim;
            sim.addReactor(react);
            setNumerics(sim);
            sim.advance(deltaT);

            CanteraGas_->getMassFractions(yTemp_.begin());

            for (size_t i = 0; i < CanteraGas_->nSpecies(); i++)
            {
                RR_[i][cellI] = (yTemp_[i] - yPre_[i]) * rhoi / deltaT;
                Qdot_[cellI] -= hc_[i] * RR_[i][cellI];
            }
        }
    }
    std::chrono::steady_clock::time_point stop_0 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_0 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_0 - start_0);
    std::cout << "beforeCUDATime = " << processingTime_0.count() << std::endl;

    // DNN

    std::chrono::steady_clock::time_point start_3 = std::chrono::steady_clock::now();

    pybind11::array_t<double> vec = pybind11::cast(inputs_);
    pybind11::module_ call_torch = pybind11::module_::import("inference_H2"); // import python file
    pybind11::object result = call_torch.attr("inference")(vec); // call function
    const double* star = result.cast<pybind11::array_t<double>>().data();

    std::vector<double> outputsVec(star, star + torch_cell.size() * 7);

    std::chrono::steady_clock::time_point stop_3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_3 - start_3);
    std::cout << "CUDATime = " << processingTime_3.count() << std::endl;

    std::chrono::steady_clock::time_point start_2 = std::chrono::steady_clock::now();
    for (size_t cellI = 0; cellI < torch_cell.size(); cellI++)
    {
        // update y
        for (size_t i = 0; i < CanteraGas_->nSpecies(); i++)
        {
            RR_[i][torch_cell[cellI]] = outputsVec[cellI * 7 + i];
            Qdot_[cellI] -= hc_[i] * RR_[i][cellI];
        }
    }
    std::chrono::steady_clock::time_point stop_2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime_2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop_2 - start_2);
    std::cout << "afterCUDATime = " << processingTime_2.count() << std::endl;

    Info << "=== end torch&ode-solve === " << endl;
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    std::cout << "allSolveTime = " << processingTime.count() << std::endl;

    return deltaTMin;
}

template <class ThermoType>
template<class DeltaTType>
//Foam::DynamicList<Foam::GpuProblem>
std::vector<Foam::GpuProblem>
Foam::dfChemistryModel<ThermoType>::getGPUProblems
(
    const DeltaTType& deltaT
)
{
    //DynamicList<GpuProblem> problemList; //single core TODO:rename it
    std::vector<GpuProblem> problemList;
    int counter = 0;
    // get cuda problemList, for all cell
    // each get problem
    forAll(T_, cellI)
    {
        
        scalar Ti = T_[cellI];
        scalar pi = p_[cellI];
        scalar rhoi = rho_[cellI];

        // set problems
        GpuProblem problem(CanteraGas_->nSpecies());
        problem.cellid = cellI;
        problem.Ti = Ti;
        problem.pi = pi/101325;
        for (size_t i = 0; i < CanteraGas_->nSpecies(); i++)
        {
            problem.Y[i] = Y_[i][cellI];
        }
        // choose DNN module
        if ((Qdot_[cellI] < Qdotact2_) && (T_[cellI] <= Tact2_)  && ( T_[cellI] >= Tact1_))//choose1
        {
            problem.DNNid = 0;
        }
        if(((Qdot_[cellI] <= Qdotact3_)&&(Qdot_[cellI] >= Qdotact2_) && (Tact3_ > T_[cellI])&&(T_[cellI] > Tact2_))||(Qdot_[cellI] > Qdotact3_))  //choose2
        {
            problem.DNNid = 1;
        }
        if  ((Qdot_[cellI] <= Qdotact3_) && (T_[cellI] >= Tact3_))//if(Ti >= Tact_))//choose3
        {
            problem.DNNid = 2;
        }
        problem.rhoi = rhoi;
        //problemList.append(problem);
        problemList.push_back(problem);
        Qdot_[cellI] = 0.0;
        counter++;
    }
    Info << "counter" << counter << endl;
    return problemList; 
}

template <class ThermoType>
void Foam::dfChemistryModel<ThermoType>::getDNNinputs
(
    //const Foam::DynamicBuffer<GpuProblem>& problemBuffer, 
    const std::vector<std::vector<GpuProblem>> problemBuffer,
    std::vector<Foam::label>& outputLength,
    std::vector<std::vector<double>>& DNNinputs, 
    std::vector<Foam::DynamicBuffer<label>>& cellIDBuffer,
    std::vector<std::vector<label>>& problemCounter
)
{
    std::vector<label> problemCounter0;     // evaluate the number of the problems of each subslave for DNN0
    std::vector<label> problemCounter1;     // evaluate the number of the problems of each subslave for DNN1
    std::vector<label> problemCounter2;     // evaluate the number of the problems of each subslave for DNN2
    std::vector<double> inputsDNN0;         // the vector constructed for inference via DNN0
    std::vector<double> inputsDNN1;         // the vector constructed for inference via DNN1
    std::vector<double> inputsDNN2;         // the vector constructed for inference via DNN2
    DynamicList<label> cellIDList0;         // store the cellID of each problem in each subslave for DNN0
    DynamicList<label> cellIDList1;         // store the cellID of each problem in each subslave for DNN1
    DynamicList<label> cellIDList2;         // store the cellID of each problem in each subslave for DNN2
    DynamicBuffer<label> cellIDList0Buffer; // store the cellIDList0 of each subslave
    DynamicBuffer<label> cellIDList1Buffer; // store the cellIDList1 of each subslave
    DynamicBuffer<label> cellIDList2Buffer; // store the cellIDList2 of each subslave

    int counter = 0;
    for (label i = 0; i < coresPerGPU; i++) // for all local core TODO: i may cause misleading
    {
        
        label counter0 = 0;
        label counter1 = 0;
        label counter2 = 0;
        //TODO: parallel the loop
        //#pragma omp parallel for schedule(dynamic) attention for multi rank, each will free once cause error!
        Info << "loop start! " << counter << endl;
        counter++;
        //Info << "problemBuffer[i].size()" << problemBuffer[i].size() << endl;
        for (label cellI = 0; cellI < problemBuffer[i].size(); cellI++) // loop coresPerGPU*problemBuffer[i].size() times
        {
            
            switch (problemBuffer[i][cellI].DNNid) //divide by Dnn id
            {
            case 0:
                try
                {
                    inputsDNN0.push_back(problemBuffer[i][cellI].rhoi);
                    inputsDNN0.push_back(problemBuffer[i][cellI].Ti);
                    inputsDNN0.push_back(problemBuffer[i][cellI].pi);
                    
                    //Info << "inputsDNN0.push_back " << problemBuffer[i][cellI].rhoi<< endl;        
                
                    for (size_t speciID = 0; speciID < CanteraGas_->nSpecies(); speciID++)
                    {
                        //Info << "problemBuffer[i][cellI].Y[speciID] " << problemBuffer[i][cellI].Y[speciID] << endl;
                        inputsDNN0.push_back(problemBuffer[i][cellI].Y[speciID]);
                    }
                    
                    counter0++;
                    //Info << "counter0" << counter0 << endl;
                    //Info << "inner append success" << endl;
                    //Info << "problemBuffer[i][cellI].cellid " << problemBuffer[i][cellI].cellid << " " << i << endl;
                    cellIDList0.append(problemBuffer[i][cellI].cellid); // store cellid for further send back
                    //Info << "DL append success" << endl;
                    break;
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                }
            case 1:
                inputsDNN1.push_back(problemBuffer[i][cellI].rhoi);
                inputsDNN1.push_back(problemBuffer[i][cellI].Ti);
                inputsDNN1.push_back(problemBuffer[i][cellI].pi);
                for (size_t speciID = 0; speciID < CanteraGas_->nSpecies(); speciID++)
                {
                    inputsDNN1.push_back(problemBuffer[i][cellI].Y[speciID]);
                }
                counter1++;
                cellIDList1.append(problemBuffer[i][cellI].cellid);
                break;

            case 2:
                inputsDNN2.push_back(problemBuffer[i][cellI].rhoi);
                inputsDNN2.push_back(problemBuffer[i][cellI].Ti);
                inputsDNN2.push_back(problemBuffer[i][cellI].pi);
                for (size_t speciID = 0; speciID < CanteraGas_->nSpecies(); speciID++)
                {
                    inputsDNN2.push_back(problemBuffer[i][cellI].Y[speciID]);
                }
                counter2++;
                cellIDList2.append(problemBuffer[i][cellI].cellid);
                break;
            
            default:
                Info<<"invalid input"<<endl;
                break;
            }
        }
        Info << "push back!" << endl;
        problemCounter0.push_back(counter0); //count number of inputs mapped to each dnn
        problemCounter1.push_back(counter1);
        problemCounter2.push_back(counter2);
        Info << "append!" << endl;
        cellIDList0Buffer.append(cellIDList0);
        cellIDList1Buffer.append(cellIDList1);
        cellIDList2Buffer.append(cellIDList2);
        Info << "clear!" << endl;
        cellIDList0.clear();
        cellIDList1.clear();
        cellIDList2.clear();
        Info << "loop complete!" << endl;
    }
    Info << "for done!" << endl;
    // get cellNumbers for each model
    label length0 = std::accumulate(problemCounter0.begin(), problemCounter0.end(), 0);
    label length1 = length0 + std::accumulate(problemCounter1.begin(), problemCounter1.end(), 0);
    label length2 = length1 + std::accumulate(problemCounter2.begin(), problemCounter2.end(), 0);

    // set output
    outputLength = {length0, length1, length2};
    DNNinputs = {inputsDNN0, inputsDNN1, inputsDNN2};
    cellIDBuffer = {cellIDList0Buffer, cellIDList1Buffer, cellIDList2Buffer};
    problemCounter = {problemCounter0, problemCounter1, problemCounter2};

    Info<<"get inputs successfully"<<endl;

    return;
}

template <class ThermoType>
void Foam::dfChemistryModel<ThermoType>::updateSolutionBuffer
(
    Foam::DynamicBuffer<Foam::GpuSolution>& solutionBuffer, 
    const double* star,
    const std::vector<Foam::label>& outputLength,
    const std::vector<Foam::DynamicBuffer<Foam::label>>& cellIDBuffer,
    std::vector<std::vector<Foam::label>>& problemCounter
)
{
    std::vector<double> outputsVec0(star, star+outputLength[0] * 21); //the float number is sample_length*sample_number
    std::vector<double> outputsVec1(star+outputLength[0] * 21, star+outputLength[1] * 21);
    std::vector<double> outputsVec2(star+outputLength[1] * 21, star+outputLength[2] * 21);

    GpuSolution solution(CanteraGas_->nSpecies());
    DynamicList<GpuSolution> solutionList; //TODO: rename

    label outputCounter0 = 0;
    label outputCounter1 = 0;
    label outputCounter2 = 0;

    for (label i = 0; i < coresPerGPU; i++) //TODO: i may cause misleading
    {
        for (size_t cellI = 0; cellI < problemCounter[0][i]; cellI++)
        {
            for (size_t speciID = 0; speciID < CanteraGas_->nSpecies(); speciID++)
            {
                solution.RRi[speciID] = outputsVec0[outputCounter0 * mixture_.nSpecies() + speciID];
            }
            solution.cellid = cellIDBuffer[0][i][cellI]; //cellid are sequential so that's fine
            solutionList.append(solution);
            outputCounter0++;
        }
        for (size_t cellI = 0; cellI < problemCounter[1][i]; cellI++)
        {
            for (size_t speciID = 0; speciID < CanteraGas_->nSpecies(); speciID++)
            {
                solution.RRi[speciID] = outputsVec1[outputCounter1 * mixture_.nSpecies() + speciID];
            }
            solution.cellid = cellIDBuffer[1][i][cellI];
            solutionList.append(solution);
            outputCounter1++;
        }
        for (size_t cellI = 0; cellI < problemCounter[2][i]; cellI++)
        {
            for (size_t speciID = 0; speciID < CanteraGas_->nSpecies(); speciID++)
            {
                solution.RRi[speciID] = outputsVec2[outputCounter2 * mixture_.nSpecies() + speciID];
            }
            solution.cellid = cellIDBuffer[2][i][cellI];
            solutionList.append(solution);
            outputCounter2++;
        }
    solutionBuffer.append(solutionList);
    solutionList.clear();
    }
    return;
}

template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::torchDCUSolve(
    const DeltaTType &deltaT)
{
    scalar deltaTMin = great;
    // set the cores slaved by a DCU
    if (!this->chemistry_)
    {
        return deltaTMin;
    }

    int localRank, localWorldSize;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localWorldSize);
    //std::cout << "localRank "<< localRank << localWorldSize<< std::endl;
    /*char message[80] = "this is no.x node";
    if(Pstream::myProcNo() == 0){
        message[11] = localRank + '0';
    }
    
    //MPI_Bcast(message, 80, MPI_CHAR, 0, localComm);
   	MPI_Bcast(message, sizeof(char)*80, MPI_BYTE, 0, localComm);
    //printf("node: %d message: %s\n",localRank,message);
    if(Pstream::myProcNo() % coresPerGPU){
        printf("node: %d message: %s\n",Pstream::myProcNo(),message); // it shows that the same memory space is
    }
    */
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Info << "=== begin torchDCUsolve === " << endl;

    /*=============================gather problems=============================*/
    //DynamicList<GpuProblem> problemList = getGPUProblems(deltaT);
    std::vector<GpuProblem> problemList = getGPUProblems(deltaT);
    //Info << "Byte of problemList " << problemList.size() << endl;
    /*==============================send problems==============================*/
    std::chrono::steady_clock::time_point start2 = std::chrono::steady_clock::now();


    //PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking); 
    if (localRank != 0) //for slave 
    {
        //Info << "problemList[0].Ti " << problemList[0].Ti << "RANK " << localRank  << endl;
        MPI_Send(problemList.data(), sizeof(GpuProblem) * problemList.size(), MPI_BYTE,0, 99, localComm); // maybe not 48 byte 复制构造函数和拷贝构造函数
        //UOPstream send((Pstream::myProcNo()/coresPerGPU)*coresPerGPU, pBufs);// sending problem to master
        //send << problemList;
    }
    //Info << "problemListSize = " << problemList.size() << endl;
    //pBufs.finishedSends();
    
    //DynamicList<GpuProblem> problemListBuffer;
    /*int sendCount[localWorldSize];       // byte number will be send
    int displs[localWorldSize];           // 接收数据在消息缓冲区中的索引
    int gatherCounter;
    
    for (int i = 0; i < localWorldSize; i++)
    {
        sendCount[i] = sizeof(GpuProblem) * problemList.size(); 
        gatherCounter += sendCount[i];
    }

    for (int i = 0; i < localWorldSize; i++)
    {
        displs[i] = localRank * sizeof(GpuProblem) * problemList.size(); 
    }

    GpuProblem *problemListBuffer = NULL;
    if(localRank == 0){

        Info << "Byte of a GpuProblem" << sizeof(GpuProblem)  << endl;
        Info << "Byte of a problemList" << sizeof(problemList)  << endl;
        Info << "Byte of a problemList.size()" << problemList.size()  << endl;
        Info << "Byte of a localWorldSize" << localWorldSize  << endl;
        Info << "Byte of problemList Buffer" << sizeof(problemListBuffer) << endl;
        problemListBuffer = (GpuProblem *)malloc(sizeof(GpuProblem) * problemList.size()*(coresPerGPU+1) * localWorldSize);
    }
    MPI_Gatherv(&problemList[0],
                sendCount[localRank],
                MPI_BYTE,
                &problemListBuffer[0],
                sendCount,
                displs,
                MPI_BYTE,
                0,
                localComm);
    if(localRank == 0){
        Info << "Byte of problemList " << sizeof(problemListBuffer) << endl;
    }*/
    
    DynamicBuffer<GpuSolution> solutionBuffer;

    std::chrono::steady_clock::time_point stop2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime2 = std::chrono::duration_cast<std::chrono::duration<double>>(stop2 - start2);
    //std::cout << "sendProblemTime = " << processingTime2.count() << std::endl;
    time_sendProblem_ += processingTime2.count();

    /*=============================submaster work start=============================*/
    if (!(Pstream::myProcNo() % coresPerGPU))
    {
        std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point start3 = std::chrono::steady_clock::now();

        label problemSize = 0; // problemSize is defined to debug
        //DynamicBuffer<GpuProblem> problemBuffer(coresPerGPU);//each submaster init a local problemBuffer TODO:rename it
        //DynamicBuffer<GpuProblem> problemBuffer(localWorldSize);
        std::vector<std::vector<GpuProblem>> problemBuffer; //!!
        //std::vector<GpuProblem> problemBuffer;
        /*==============================gather problems==============================*/
        problemBuffer.push_back(problemList);
        Info << "problem1 size = " << problemList[0].Y[0] << endl;
        //problemBuffer[0] = problemList; //problemList of submaster get index 0
        problemSize += problemBuffer[0].size();
        //Info << "problemSize = " << problemSize << endl;
    
        //#pragma omp parallel for 
        //for (label i = 1; i < coresPerGPU; i++)
       
        for (label i = 1; i < localWorldSize; i++)
        {
            MPI_Status status;
            MPI_Probe(i, 99, localComm, &status);
            int byteNumber;
            MPI_Get_count(&status, MPI_BYTE, &byteNumber);
            //std::cout << "byteNumber = " << byteNumber << " " << byteNumber/sizeof(GpuProblem) << std::endl;
            std::vector<GpuProblem> recv_buffer(byteNumber/sizeof(GpuProblem));
             
            //std::cout << "sendProblemSize = " << sendProblemSize[i] << std::endl;
            MPI_Recv(recv_buffer.data(), byteNumber, MPI_BYTE, i, 99, localComm, &status); // the recv buffer is longer than send buffer
            Info << "Recv ok" << endl;
            //UIPstream recv(i + Pstream::myProcNo(), pBufs);
            //recv >> problemBuffer[i];  //recv previous send problem and append to problemList
            //Info << "recvList[0].Ti "<< recv_buffer[0].Y[0] << "RANK " << i <<endl; //get problem!
            Info << "recv_buffer = " << recv_buffer.size() << endl;
            problemBuffer.emplace_back(std::move(recv_buffer));
            Info << "problemSize = " << problemBuffer.size() << endl;

            problemSize += problemBuffer[i].size();
            Info << "problemSize = " << i << problemSize << endl;
            
        }
        Info << "problemBuffer size = " << problemBuffer.size() << endl;
        //Info << "problem1 size = " << problemBuffer[1][0].Y[0] << endl;
        //Info << "problemSize = " << problemSize << endl;

        std::chrono::steady_clock::time_point stop3 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime3 = std::chrono::duration_cast<std::chrono::duration<double>>(stop3 - start3);
        //std::cout << "RecvProblemTime = " << processingTime3.count() << std::endl;
        time_RecvProblem_ += processingTime3.count();

        /*==============================construct DNN inputs==============================*/
        std::vector<label> outputLength;
        std::vector<std::vector<double>> DNNinputs;     // vectors for the inference of DNN
        std::vector<DynamicBuffer<label>> cellIDBuffer; // Buffer contains the cell numbers
        std::vector<std::vector<label>> problemCounter; // evaluate the number of the problems of each subslave

        std::chrono::steady_clock::time_point start5 = std::chrono::steady_clock::now();
        getDNNinputs(problemBuffer, outputLength, DNNinputs, cellIDBuffer, problemCounter);
        std::chrono::steady_clock::time_point stop5 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime5 = std::chrono::duration_cast<std::chrono::duration<double>>(stop5 - start5);
        //std::cout << "getDNNinputsTime = " << processingTime5.count() << std::endl;
        time_getDNNinputs_ += processingTime5.count();

        /*=============================inference via pybind11=============================*/
        std::chrono::steady_clock::time_point start7 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point start8 = std::chrono::steady_clock::now();

        /*pybind11::array_t<double> vec0 = pybind11::cast(DNNinputs[0]); // cast vector to np.array
        pybind11::array_t<double> vec1 = pybind11::cast(DNNinputs[1]);
        pybind11::array_t<double> vec2 = pybind11::cast(DNNinputs[2]);*/
        pybind11::array_t<double> vec0 = pybind11::array_t<double>({DNNinputs[0].size()}, {8}, &DNNinputs[0][0]);
        pybind11::array_t<double> vec1 = pybind11::array_t<double>({DNNinputs[1].size()}, {8}, &DNNinputs[1][0]);
        pybind11::array_t<double> vec2 = pybind11::array_t<double>({DNNinputs[2].size()}, {8}, &DNNinputs[2][0]);

        std::chrono::steady_clock::time_point stop8 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime8 = std::chrono::duration_cast<std::chrono::duration<double>>(stop8 - start8);
        //std::cout << "vec2ndarrayTime = " << processingTime8.count() << std::endl;
        time_vec2ndarray_ += processingTime8.count();

        pybind11::module_ call_torch = pybind11::module_::import("inference2"); // import python file

        std::chrono::steady_clock::time_point start9 = std::chrono::steady_clock::now();

        pybind11::object result = call_torch.attr("inference")(vec0, vec1, vec2); // call python function
        const double* star = result.cast<pybind11::array_t<double>>().data();

        std::chrono::steady_clock::time_point stop9 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime9 = std::chrono::duration_cast<std::chrono::duration<double>>(stop9 - start9);
        //std::cout << "pythonTime = " << processingTime9.count() << std::endl;
        time_python_ += processingTime9.count();

        std::chrono::steady_clock::time_point stop7 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime7 = std::chrono::duration_cast<std::chrono::duration<double>>(stop7 - start7);
        //std::cout << "DNNinferenceTime = " << processingTime7.count() << std::endl;
        time_DNNinference_ += processingTime7.count();

        /*=============================construct solutions=============================*/
        std::chrono::steady_clock::time_point start6 = std::chrono::steady_clock::now();
        updateSolutionBuffer(solutionBuffer, star, outputLength, cellIDBuffer, problemCounter);
        std::chrono::steady_clock::time_point stop6 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime6 = std::chrono::duration_cast<std::chrono::duration<double>>(stop6 - start6);
        //std::cout << "updateSolutionBufferTime = " << processingTime6.count() << std::endl;
        time_updateSolutionBuffer_ += processingTime6.count();

        std::chrono::steady_clock::time_point stop1 = std::chrono::steady_clock::now();
        std::chrono::duration<double> processingTime1 = std::chrono::duration_cast<std::chrono::duration<double>>(stop1 - start1);
        //std::cout << "submasterTime = " << processingTime1.count() << std::endl;  
        time_submaster_ += processingTime1.count();
    }

    /*=============================send and recv solutions=============================*/
    std::chrono::steady_clock::time_point start4 = std::chrono::steady_clock::now();

    DynamicList<GpuSolution> finalList;
    PstreamBuffers pBufs2(Pstream::commsTypes::nonBlocking);
    if (!(Pstream::myProcNo() % coresPerGPU))
    {
        finalList = solutionBuffer[0];
        //#pragma omp parallel for 
        for (label i = 1; i < coresPerGPU; i++)
        {
            UOPstream send(i + Pstream::myProcNo(), pBufs2);
            send << solutionBuffer[i];
        }
    }
    pBufs2.finishedSends();
    if (Pstream::myProcNo() % coresPerGPU)
    {
        UIPstream recv((Pstream::myProcNo()/coresPerGPU)*coresPerGPU, pBufs2);
        recv >> finalList;
    }

    std::chrono::steady_clock::time_point stop4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime4 = std::chrono::duration_cast<std::chrono::duration<double>>(stop4 - start4);
    //std::cout << "SendRecvSolutionTime = " << processingTime4.count() << std::endl;
    time_sendRecvSolution_ += processingTime4.count();

    /*=============================update RR fields=============================*/
    for (size_t cellI = 0; cellI < finalList.size(); cellI++)
    {
        for (size_t speciID = 0; speciID < CanteraGas_->nSpecies(); speciID++)
        {
            RR_[speciID][finalList[cellI].cellid] = finalList[cellI].RRi[speciID];
            Qdot_[finalList[cellI].cellid] -= hc_[speciID] * RR_[speciID][finalList[cellI].cellid];
        }
    }

    Info << "=== end torch&ode-CUDAsolve === " << endl;

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    //std::cout << "allSolveTime = " << processingTime.count() << std::endl;
    time_allsolve_ += processingTime.count();

    return deltaTMin;
}


template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::correctThermo()
{
    forAll(T_, celli)
    {
        forAll(Y_, i)
        {
            yTemp_[i] = Y_[i][celli];
        }
        CanteraGas_->setState_PY(p_[celli], yTemp_.begin());
        CanteraGas_->setState_HP(thermo_.he()[celli], p_[celli]); // setState_HP needs (J/kg)

        T_[celli] = CanteraGas_->temperature();

        psi_[celli] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT(); // meanMolecularWeight() kg/kmol    RT() Joules/kmol

        mu_[celli] = mixture_.CanteraTransport()->viscosity(); // Pa-s

        alpha_[celli] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass()); // kg/(m*s)
        // thermalConductivity() W/m/K
        // cp_mass()   J/kg/K


        if (mixture_.transportModelName() == "UnityLewis")
        {
            forAll(rhoD_, i)
            {
                rhoD_[i][celli] = alpha_[celli];
            }
        }
        else
        {
            mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin()); // m2/s

            CanteraGas_->getEnthalpy_RT(hrtTemp_.begin()); //hrtTemp_=m_h0_RT non-dimension
            // constant::physicoChemical::R.value()   J/(mol·k)
            const scalar RT = constant::physicoChemical::R.value()*1e3*T_[celli]; // J/kmol/K
            forAll(rhoD_, i)
            {
                rhoD_[i][celli] = rho_[celli]*dTemp_[i];

                // CanteraGas_->molecularWeight(i)    kg/kmol
                hai_[i][celli] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
            }
        }
    }


    const volScalarField::Boundary& pBf = p_.boundaryField();

    const volScalarField::Boundary& rhoBf = rho_.boundaryField();

    volScalarField::Boundary& TBf = T_.boundaryFieldRef();

    volScalarField::Boundary& psiBf = psi_.boundaryFieldRef();

    volScalarField::Boundary& hBf = thermo_.he().boundaryFieldRef();

    volScalarField::Boundary& muBf = mu_.boundaryFieldRef();

    volScalarField::Boundary& alphaBf = alpha_.boundaryFieldRef();

    forAll(T_.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = pBf[patchi];
        const fvPatchScalarField& prho = rhoBf[patchi];
        fvPatchScalarField& pT = TBf[patchi];
        fvPatchScalarField& ppsi = psiBf[patchi];
        fvPatchScalarField& ph = hBf[patchi];
        fvPatchScalarField& pmu = muBf[patchi];
        fvPatchScalarField& palpha = alphaBf[patchi];

        if (pT.fixesValue())
        {
            forAll(pT, facei)
            {
                forAll(Y_, i)
                {
                    yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
                }
                CanteraGas_->setState_TPY(pT[facei], pp[facei], yTemp_.begin());

                ph[facei] = CanteraGas_->enthalpy_mass();

                ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();

                pmu[facei] = mixture_.CanteraTransport()->viscosity();

                palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());
                if (mixture_.transportModelName() == "UnityLewis")
                {
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                    }
                }
                else
                {
                    mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin());

                    CanteraGas_->getEnthalpy_RT(hrtTemp_.begin());
                    const scalar RT = constant::physicoChemical::R.value()*1e3*pT[facei];
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = prho[facei]*dTemp_[i];

                        hai_[i].boundaryFieldRef()[patchi][facei] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
                    }
                }
            }
        }
        else
        {
            forAll(pT, facei)
            {
                forAll(Y_, i)
                {
                    yTemp_[i] = Y_[i].boundaryField()[patchi][facei];
                }
                CanteraGas_->setState_PY(pp[facei], yTemp_.begin());
                CanteraGas_->setState_HP(ph[facei], pp[facei]);

                pT[facei] = CanteraGas_->temperature();

                ppsi[facei] = CanteraGas_->meanMolecularWeight()/CanteraGas_->RT();

                pmu[facei] = mixture_.CanteraTransport()->viscosity();

                palpha[facei] = mixture_.CanteraTransport()->thermalConductivity()/(CanteraGas_->cp_mass());

                if (mixture_.transportModelName() == "UnityLewis")
                {
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = palpha[facei];
                    }
                }
                else
                {
                    mixture_.CanteraTransport()->getMixDiffCoeffsMass(dTemp_.begin());

                    CanteraGas_->getEnthalpy_RT(hrtTemp_.begin());
                    const scalar RT = constant::physicoChemical::R.value()*1e3*pT[facei];
                    forAll(rhoD_, i)
                    {
                        rhoD_[i].boundaryFieldRef()[patchi][facei] = dTemp_[i];

                        hai_[i].boundaryFieldRef()[patchi][facei] = hrtTemp_[i]*RT/CanteraGas_->molecularWeight(i);
                    }
                }
            }
        }
    }
}

template<class ThermoType>
void Foam::dfChemistryModel<ThermoType>::solveSingle
(
    ChemistryProblem& problem, ChemistrySolution& solution
)
{

    // Timer begins
    clockTime time;
    time.timeIncrement();

    Cantera::Reactor react;
    const scalar Ti = problem.Ti;
    const scalar pi = problem.pi;
    const scalar rhoi = problem.rhoi;
    const scalarList yPre_ = problem.Y;

    CanteraGas_->setState_TPY(Ti, pi, yPre_.begin());

    react.insert(mixture_.CanteraSolution());
    react.setEnergy(0); // keep T const before and after sim.advance. this will give you a little improvement
    Cantera::ReactorNet sim;
    sim.addReactor(react);
    setNumerics(sim);

    sim.advance(problem.deltaT);

    CanteraGas_->getMassFractions(yTemp_.begin());

    for (size_t i=0; i<CanteraGas_->nSpecies(); i++)
    {
        solution.RRi[i] = (yTemp_[i] - yPre_[i]) / problem.deltaT * rhoi;
    }

    // Timer ends
    solution.cpuTime = time.timeIncrement();

    solution.cellid = problem.cellid;
}



template <class ThermoType>
template<class DeltaTType>
Foam::DynamicList<Foam::ChemistryProblem>
Foam::dfChemistryModel<ThermoType>::getProblems
(
    const DeltaTType& deltaT
)
{
    const scalarField& T = T_;
    const scalarField& p = p_;
    const scalarField& rho = rho_;


    DynamicList<ChemistryProblem> solved_problems(p.size(), ChemistryProblem(mixture_.nSpecies()));

    forAll(T, celli)
    {
        {
            for(label i = 0; i < mixture_.nSpecies(); i++)
            {
                yTemp_[i] = Y_[i][celli];
            }

            CanteraGas_->setState_TPY(T[celli], p[celli], yTemp_.begin());
            CanteraGas_->getConcentrations(cTemp_.begin());

            ChemistryProblem problem;
            problem.Y = yTemp_;
            problem.Ti = T[celli];
            problem.pi = p[celli];
            problem.rhoi = rho_[celli];
            problem.deltaT = deltaT[celli];
            problem.cpuTime = cpuTimes_[celli];
            problem.cellid = celli;

            solved_problems[celli] = problem;
        }
        
    }

    return solved_problems;
}


template <class ThermoType>
Foam::DynamicList<Foam::ChemistrySolution>
Foam::dfChemistryModel<ThermoType>::solveList
(
    UList<ChemistryProblem>& problems
)
{
    DynamicList<ChemistrySolution> solutions(
        problems.size(), ChemistrySolution(mixture_.nSpecies()));

    for(label i = 0; i < problems.size(); ++i)
    {
        solveSingle(problems[i], solutions[i]);
    }
    return solutions;
}


template <class ThermoType>
Foam::RecvBuffer<Foam::ChemistrySolution>
Foam::dfChemistryModel<ThermoType>::solveBuffer
(
    RecvBuffer<ChemistryProblem>& problems
)
{
    // allocate the solutions buffer
    RecvBuffer<ChemistrySolution> solutions;

    for(auto& p : problems)
    {
        solutions.append(solveList(p));
    }
    return solutions;
}



template <class ThermoType>
Foam::scalar
Foam::dfChemistryModel<ThermoType>::updateReactionRates
(
    const RecvBuffer<ChemistrySolution>& solutions
)
{
    scalar deltaTMin = great;

    for(const auto& array : solutions)
    {
        for(const auto& solution : array)
        {

            for(label j = 0; j < mixture_.nSpecies(); j++)
            {
                this->RR_[j][solution.cellid] = solution.RRi[j];
            }

            cpuTimes_[solution.cellid] = solution.cpuTime;
        }
    }

    return deltaTMin;
}



template <class ThermoType>
Foam::LoadBalancer
Foam::dfChemistryModel<ThermoType>::createBalancer()
{
    const IOdictionary chemistryDict_tmp
        (
            IOobject
            (
                "CanteraTorchProperties",
                thermo_.db().time().constant(),
                thermo_.db(),
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        );

    return LoadBalancer(chemistryDict_tmp);
}



template <class ThermoType>
template <class DeltaTType>
Foam::scalar Foam::dfChemistryModel<ThermoType>::solve_loadBalance
(
    const DeltaTType& deltaT
)
{
    Info<<"=== begin DLB-solve === "<<endl;
    // CPU time analysis
    clockTime timer;
    scalar t_getProblems(0);
    scalar t_updateState(0);
    scalar t_balance(0);
    scalar t_solveBuffer(0);
    scalar t_unbalance(0);

    if(!this->chemistry_)
    {
        return great;
    }

    timer.timeIncrement();
    DynamicList<ChemistryProblem> allProblems = getProblems(deltaT);
    t_getProblems = timer.timeIncrement();

    RecvBuffer<ChemistrySolution> incomingSolutions;

    if(balancer_.active())
    {
        timer.timeIncrement();
        balancer_.updateState(allProblems);
        t_updateState = timer.timeIncrement();

        timer.timeIncrement();
        auto guestProblems = balancer_.balance(allProblems);
        auto ownProblems = balancer_.getRemaining(allProblems);
        t_balance = timer.timeIncrement();

        timer.timeIncrement();
        auto ownSolutions = solveList(ownProblems);
        auto guestSolutions = solveBuffer(guestProblems);
        t_solveBuffer = timer.timeIncrement();

        timer.timeIncrement();
        incomingSolutions = balancer_.unbalance(guestSolutions);
        incomingSolutions.append(ownSolutions);
        t_unbalance = timer.timeIncrement();
    }
    else
    {
        timer.timeIncrement();
        incomingSolutions.append(solveList(allProblems));
        t_solveBuffer = timer.timeIncrement();
    }

    if(balancer_.log())
    {
        balancer_.printState();
        cpuSolveFile_() << setw(22)
                        << this->time().timeOutputValue()<<tab
                        << setw(22) << t_getProblems<<tab
                        << setw(22) << t_updateState<<tab
                        << setw(22) << t_balance<<tab
                        << setw(22) << t_solveBuffer<<tab
                        << setw(22) << t_unbalance<<tab
                        << setw(22) << Pstream::myProcNo()
                        << endl;
    }

    Info<<"=== end DLB-solve === "<<endl;
    return updateReactionRates(incomingSolutions);
}

// ************************************************************************* //
