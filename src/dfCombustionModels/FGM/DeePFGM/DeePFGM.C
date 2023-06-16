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

\*---------------------------------------------------------------------------*/

#include "DeePFGM.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::DeePFGM<ReactionThermo>::DeePFGM
(
    const word& modelType,
    ReactionThermo& thermo,
    const compressibleTurbulenceModel& turb,
    const word& combustionProperties
)
:
    baseFGM<ReactionThermo>(modelType, thermo, turb, combustionProperties),
    tableSolver(
                 baseFGM<ReactionThermo>::speciesNames_,
                 baseFGM<ReactionThermo>::scaledPV_,
                 baseFGM<ReactionThermo>::flameletT_,
                 baseFGM<ReactionThermo>::Ycmaxall_
               )
{
    //- retrieval data from table
    PyObject* module=initialize_module();
    PyObject* func=initialize_function(module);
    retrieval(module,func);
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class ReactionThermo>
Foam::combustionModels::DeePFGM<ReactionThermo>::~DeePFGM()
{}

template<class ReactionThermo>
PyObject* Foam::combustionModels::DeePFGM<ReactionThermo>::initialize_module()
{
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        std::cout << "python init failed" << std::endl;
    }
    // 2、初始化python系统文件路径，保证可以访问到 .py文件
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/home/whx/deepflame-dev/src/dfCombustionModels/FGM/DeePFGM/FGMinference')");
    PyObject* module = PyImport_ImportModule("inference");
    if (module == nullptr)
    {
        std::cout <<"module not found: inference" << std::endl;
    }
    return module;
}

template<class ReactionThermo>
PyObject* Foam::combustionModels::DeePFGM<ReactionThermo>::initialize_function(PyObject* module)
{
    PyObject* func = PyObject_GetAttrString(module, "FGM");
    if (!func || !PyCallable_Check(func))
    {
        std::cout <<"function not found: FGM" << std::endl;
    }
    else
    {
        cout <<"function  found: FGM" << std::endl;
    }
    return func;
}

template<class ReactionThermo>
void Foam::combustionModels::DeePFGM<ReactionThermo>::correct()
{
    //- initialize flame kernel
    baseFGM<ReactionThermo>::initialiseFalmeKernel();

    //- solve transport equation
    baseFGM<ReactionThermo>::transport();

    //update enthalpy using lookup data
    if(!(this->solveEnthalpy_))
    {
        this->He_ = this->Z_*(H_fuel-H_ox) + H_ox;
    }
    PyObject* module=initialize_module();
    PyObject* func=initialize_function(module);
    //- retrieval data from table
    retrieval(module,func);
    
}
template<class ReactionThermo>
int Foam::combustionModels::DeePFGM<ReactionThermo>::prediction
    (
        double z_s[], double c_s[], double gz_s[], double gc_s[],
        double gzc_s[],  int phinum, int dimension,double* result,int size,PyObject* module,PyObject* func
    )
    {
        PyObject* py_phinum = PyLong_FromLong(phinum);
        PyObject* py_dimension = PyLong_FromLong(dimension);
        // 将C++的数组参数转换为Python列表对象
        PyObject* py_z = PyList_New(size);
        PyObject* py_c = PyList_New(size);
        PyObject* py_gz = PyList_New(size);
        PyObject* py_gc = PyList_New(size);
        PyObject* py_gzc = PyList_New(size);
        for (int i = 0; i < size; i++) {
            PyList_SET_ITEM(py_z, i, PyFloat_FromDouble(z_s[i]));
            PyList_SET_ITEM(py_c, i, PyFloat_FromDouble(c_s[i]));
            PyList_SET_ITEM(py_gz, i, PyFloat_FromDouble(gz_s[i]));
            PyList_SET_ITEM(py_gc, i, PyFloat_FromDouble(gc_s[i]));
            PyList_SET_ITEM(py_gzc, i, PyFloat_FromDouble(gzc_s[i]));
        }
        PyObject* args = PyTuple_Pack(7, py_z, py_c,py_gz,py_gc,py_gzc,py_phinum,py_dimension);
        PyObject* py_result = PyObject_CallObject(func, args);
        // 5、调用函数
        // PyObject_CallObject(func, nullptr);
        if (PyList_Check(py_result)) {
            for (int i = 0; i < size; i++) {
                PyObject* item = PyList_GetItem(py_result, i);
                double value = PyFloat_AsDouble(item);
                result[i] = value;
                // Info<<result[i]<<endl;
            }
        }
        else
        {
            Info<<"It is not a list"<<endl;
        }
        // 6、结束python接口初始化
        Py_DECREF(py_z);
        Py_DECREF(py_c);
        Py_DECREF(py_gz);
        Py_DECREF(py_gc);
        Py_DECREF(py_gzc);
        Py_DECREF(args);
        Py_DECREF(py_result);
        // Py_DECREF(func);
        // Py_DECREF(module);
        // Py_Finalize();
        return 0;
                
    }
template<class ReactionThermo>
void Foam::combustionModels::DeePFGM<ReactionThermo>::retrieval(PyObject* module,PyObject* func)
{

    tmp<volScalarField> tk(this->turbulence().k());
    volScalarField& k = const_cast<volScalarField&>(tk());
    scalarField& kCells =k.primitiveFieldRef();

    tmp<volScalarField> tepsilon(this->turbulence().epsilon());
     volScalarField& epsilon = const_cast<volScalarField&>(tepsilon());
    const scalarField& epsilonCells =epsilon.primitiveFieldRef();

    tmp<volScalarField> tmu = this->turbulence().mu();  
    volScalarField& mu = const_cast<volScalarField&>(tmu());
    scalarField& muCells = mu.primitiveFieldRef();     
    int dim=2;
    // Info<< "pause1" << endl;
    //- calculate reacting flow solution
    const scalar Zl{z_Tb5[0]};  
    const scalar Zr{z_Tb5[NZL-1]};
 

    dimensionedScalar TMin("TMin",dimensionSet(0,0,0,1,0,0,0),200.0);    
    dimensionedScalar TMax("TMax",dimensionSet(0,0,0,1,0,0,0),3000.0);  

    forAll(this->rho_, celli)  
    {

        this->chi_ZCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli] *this->ZvarCells_[celli]; 

        this->chi_ZcCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli] *this->ZcvarCells_[celli]; 

        if(this->ZCells_[celli] >= Zl && this->ZCells_[celli] <= Zr     
        && this->combustion_ && this->cCells_[celli] > this->small) 
        {

            double kc_s = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],kctau_Tb5);      
            double tau = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],tau_Tb5);    
            double sl = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],sl_Tb5);     
            double dl = this->lookup1d(NZL,z_Tb5,this->ZCells_[celli],th_Tb5);   

            this->chi_cCells_[celli] =
                this->RANSsdrFLRmodel(this->cvarCells_[celli],epsilonCells[celli],
                    kCells[celli],muCells[celli]/this->rho_[celli],
                    sl,dl,tau,kc_s,this->rho_[celli]);  

        }
        else
        {
            this->chi_cCells_[celli] = 1.0*epsilonCells[celli]/kCells[celli]*this->cvarCells_[celli]; 
        }
    }
    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    int Ncells = this->chi_cCells_.size();
    /*new double[Ncells]{}*/
    double gz_s[Ncells]={};
    double gcz_s[Ncells]={};
    double Ycmax_s[Ncells]={};
    double gc_s[Ncells]={};
    double z_s[Ncells]={};
    double Wt_s[Ncells]={};
    double mu_s[Ncells]={};
    double c_s[Ncells]={};

    // double gz{cal_gvar(this->ZCells_[celli],this->ZvarCells_[celli])};   
    // double gcz{cal_gcor(this->ZCells_[celli],this->cCells_[celli],this->ZvarCells_[celli],this->cvarCells_[celli],this->ZcvarCells_[celli])},
    //         Ycmax{-1.0},cNorm{},gc{};    

    forAll(this->rho_, celli)
    {
        z_s[celli]=this->ZCells_[celli];
        gz_s[celli]=cal_gvar(this->ZCells_[celli],this->ZvarCells_[celli]);
        gcz_s[celli]=cal_gcor(this->ZCells_[celli],this->cCells_[celli],this->ZvarCells_[celli],this->cvarCells_[celli],this->ZcvarCells_[celli]);
        gc_s[celli]=cal_gvar(this->cCells_[celli],this->cvarCells_[celli],Ycmax_s[0]);
    }
    if(scaledPV_)
    {
        forAll(this->rho_, celli)
        {
            c_s[celli]=this->cCells_[celli];
        }
        // Info<<"Sclaed PV"<<endl;
    } 

    // if(scaledPV_)    
    // {
    //     cNorm = this->cCells_[celli];  
    // }
    // else
    // {
    //     Ycmax = this->lookup5d(NZ,z_Tb3,this->ZCells_[celli],
    //                           NC,c_Tb3,0.0,
    //                           NGZ,gz_Tb3,gz,
    //                           NGC,gc_Tb3,0.0,
    //                           NZC,gzc_Tb3,0.0,
    //                           Ycmax_Tb3);    
    //     Ycmax = max(this->smaller,Ycmax);    
    //     cNorm = this->cCells_[celli]/Ycmax; 
    // }

    // gc = cal_gvar(this->cCells_[celli],this->cvarCells_[celli],Ycmax);  

    prediction(z_s,c_s,gz_s,gc_s,gcz_s,4,dim,Wt_s,Ncells,module,func);
    prediction(z_s,c_s,gz_s,gc_s,gcz_s,7,dim,mu_s,Ncells,module,func);
    // Info<< "pause4" << endl;
    forAll(this->rho_, celli)
    {
        this->WtCells_[celli]=Wt_s[celli];
        muCells[celli]=mu_s[celli];
    }   

   // -------------------- Yis begin ------------------------------
    if(NY > 0)
    {
        double YH2O_s[Ncells]={};
        double YCO_s[Ncells]={};
        double YCO2_s[Ncells]={};
        prediction(z_s,c_s,gz_s,gc_s,gcz_s,8,dim,YH2O_s,Ncells,module,func);
        prediction(z_s,c_s,gz_s,gc_s,gcz_s,9,dim,YCO_s,Ncells,module,func);
        prediction(z_s,c_s,gz_s,gc_s,gcz_s,10,dim,YCO2_s,Ncells,module,func);
        forAll(this->rho_, celli)
        {
            this->YH2OCells_[celli]=YH2O_s[celli];
            this->YCOCells_[celli]=YCO_s[celli];
            this->YCO2Cells_[celli]=YCO2_s[celli];
        }
    }

    // -------------------- Yis end ------------------------------

    double omegac_s[Ncells]={};
    double comegac_s[Ncells]={};
    double zomegac_s[Ncells]={};
    prediction(z_s,c_s,gz_s,gc_s,gcz_s,0,dim,omegac_s,Ncells,module,func);
    prediction(z_s,c_s,gz_s,gc_s,gcz_s,1,dim,comegac_s,Ncells,module,func);
    prediction(z_s,c_s,gz_s,gc_s,gcz_s,2,dim,zomegac_s,Ncells,module,func);
    forAll(this->rho_, celli)
    {
        this->omega_cCells_[celli]=(omegac_s[celli]+ (
                  scaledPV_
                  ? this->chi_ZCells_[celli]*this->cCells_[celli]
                    *lookup2d(NZ,z_Tb3,this->ZCells_[celli],
                                   NGZ,gz_Tb3,gz_s[celli],d2Yeq_Tb2)
                  : 0.0
              ))*this->rho_[celli];
        this->cOmega_cCells_[celli]=comegac_s[celli]*this->rho_[celli];
        this->ZOmega_cCells_[celli]=zomegac_s[celli]*this->rho_[celli];
    }
    forAll(this->rho_, celli)
    {   
        if(this->ZCells_[celli] <= Zl || this->ZCells_[celli] >= Zr
       || !this->combustion_ || this->cCells_[celli] <= this->small) 
        {
        this->omega_cCells_[celli] = 0.0;
        this->cOmega_cCells_[celli] = 0.0;  
        this->ZOmega_cCells_[celli] = 0.0; 
        }
    }

    if(flameletT_)   
    {
        double T_s[Ncells]={};
        prediction(z_s,c_s,gz_s,gc_s,gcz_s,6,dim,T_s,Ncells,module,func);
        forAll(this->rho_, celli)
        {
            this->TCells_[celli]=T_s[celli];
        }
    }
    else
    {
        double Cp_s[Ncells]={};
        double Hf_s[Ncells]={};
        prediction(z_s,c_s,gz_s,gc_s,gcz_s,3,dim,Cp_s,Ncells,module,func);
        prediction(z_s,c_s,gz_s,gc_s,gcz_s,5,dim,Hf_s,Ncells,module,func);
        forAll(this->rho_, celli)
        { 
            this->CpCells_[celli]=Cp_s[celli];
            this->HfCells_[celli]=Hf_s[celli];
            this->TCells_[celli] = (this->HCells_[celli]-this->HfCells_[celli])/this->CpCells_[celli]
                        + this->T0;
        }
    }


    //-----------update boundary---------------------------------

    forAll(this->rho_.boundaryFieldRef(), patchi)   
    {
        fvPatchScalarField& pZ = this->Z_.boundaryFieldRef()[patchi];     
        fvPatchScalarField& pZvar = this->Zvar_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pH = this->He_.boundaryFieldRef()[patchi];     
        fvPatchScalarField& pWt = this->Wt_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pCp = this->Cp_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pHf = this->Hf_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pT = this->T_.boundaryFieldRef()[patchi];     
        fvPatchScalarField& prho_ = this->rho_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pomega_c = this->omega_c_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pcOmega_c = this->cOmega_c_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pZOmega_c = this->ZOmega_c_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pc = this->c_.boundaryFieldRef()[patchi];          
        fvPatchScalarField& pcvar = this->cvar_.boundaryFieldRef()[patchi];    
        fvPatchScalarField& pZcvar = this->Zcvar_.boundaryFieldRef()[patchi];    
        fvPatchScalarField& pchi_Z = this->chi_Z_.boundaryFieldRef()[patchi];  
        fvPatchScalarField& pchi_c = this->chi_c_.boundaryFieldRef()[patchi];   
        fvPatchScalarField& pchi_Zc = this->chi_Zc_.boundaryFieldRef()[patchi];   

        tmp<scalarField> tmuw = this->turbulence().mu(patchi);
        scalarField& pmu = const_cast<scalarField&>(tmuw());    

        fvPatchScalarField& pk = k.boundaryFieldRef()[patchi];
        fvPatchScalarField& pepsilon = epsilon.boundaryFieldRef()[patchi];


        forAll(prho_, facei)   
        {

            pchi_Z[facei] = 1.0*pepsilon[facei]/pk[facei] *pZvar[facei]; 

            pchi_Zc[facei]= 1.0*pepsilon[facei]/pk[facei] *pZcvar[facei]; 

            if(pZ[facei] >= Zl && pZ[facei] <= Zr
            && this->combustion_ && pc[facei] > this->small) 
            {

                double kc_s = lookup1d(NZL,z_Tb5,pZ[facei],kctau_Tb5);     
                double tau = lookup1d(NZL,z_Tb5,pZ[facei],tau_Tb5);    
                double sl = lookup1d(NZL,z_Tb5,pZ[facei],sl_Tb5);      
                double dl = lookup1d(NZL,z_Tb5,pZ[facei],th_Tb5);      

                pchi_c[facei] =
                    RANSsdrFLRmodel(pcvar[facei],pepsilon[facei],
                        pk[facei],pmu[facei]/prho_[facei],
                        sl,dl,tau,kc_s,prho_[facei]);
            }
            else
            {      
                pchi_c[facei] = 1.0*pepsilon[facei]/pk[facei] *pcvar[facei]; 

            }
        }
         // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

        int Npatch=pZ.size();
        double z_t[Npatch]={};
        double gz_t[Npatch]={};
        double gcz_t[Npatch]={};
        double c_t[Npatch]={};
        double gc_t[Npatch]={};
        double Wt_t[Npatch]={};
        double mu_t[Npatch]={};
        forAll(prho_, facei)
        {
            z_t[facei]=pZ[facei];
            gz_t[facei]=cal_gvar(pZ[facei],pZvar[facei]);
            gcz_t[facei]=cal_gcor(pZ[facei],pc[facei],pZvar[facei],pcvar[facei],pZcvar[facei]);
            gc_t[facei]=cal_gvar(pc[facei],pcvar[facei],Ycmax_s[0]);

        }
        if(scaledPV_)
        {
            forAll(prho_, facei)
            {
                c_t[facei]=pc[facei];
            }
        }
        prediction(z_t,c_t,gz_t,gc_t,gcz_t,4,dim,Wt_t,Npatch,module,func);
        prediction(z_t,c_t,gz_t,gc_t,gcz_t,7,dim,mu_t,Npatch,module,func);
        forAll(prho_, facei)
        {
            pWt[facei]=Wt_t[facei];
            pmu[facei]=mu_t[facei];
        }
        double omegac_t[Npatch]={};
        double comegac_t[Npatch]={};
        double zomegac_t[Npatch]={};
        prediction(z_t,c_t,gz_t,gc_t,gcz_t,0,dim,omegac_t,Npatch,module,func);
        prediction(z_t,c_t,gz_t,gc_t,gcz_t,1,dim,comegac_t,Npatch,module,func);
        prediction(z_t,c_t,gz_t,gc_t,gcz_t,2,dim,zomegac_t,Npatch,module,func);
        forAll(prho_, facei)
        {
            pomega_c[facei]=(omegac_t[facei]+ (
                    scaledPV_
                    ? pchi_Z[facei]*pc[facei]
                        *lookup2d(NZ,z_Tb3,pZ[facei],
                                    NGZ,gz_Tb3,gz_t[facei],d2Yeq_Tb2)
                    : 0.0
                ))*prho_[facei];
            pcOmega_c[facei]=comegac_t[facei]*prho_[facei];
            pZOmega_c[facei]=zomegac_t[facei]*prho_[facei];
        }
        forAll(prho_, facei)
        {   
            if(pZ[facei] <= Zl || pZ[facei] >= Zr
            || !this->combustion_ || pc[facei]  <= this->small) 
            {
                pomega_c[facei] = 0.0;
                pcOmega_c[facei] = 0.0;  
                pZOmega_c[facei] = 0.0; 
            }
        }
            if(flameletT_)   
            {
                double T_t[Npatch]={};
                prediction(z_t,c_t,gz_t,gc_t,gcz_t,6,dim,T_t,Npatch,module,func);
                forAll(prho_, facei)
                { 
                    pT[facei]=T_t[facei];
                }
            }
            else
            {
                double Cp_t[Npatch]={};
                double Hf_t[Npatch]={};
                prediction(z_t,c_t,gz_t,gc_t,gcz_t,3,dim,Cp_t,Npatch,module,func);
                prediction(z_t,c_t,gz_t,gc_t,gcz_t,5,dim,Hf_t,Npatch,module,func);
                forAll(prho_, facei)
                { 
                    pCp[facei]=Cp_t[facei];
                    pHf[facei]=Hf_t[facei];
                    pT[facei] = (pH[facei]-pHf[facei])/pCp[facei]
                                + this->T0;
                }
            }
    this->T_.max(TMin);  
    this->T_.min(TMax);  

    if(this->mesh().time().timeIndex() > 0)   
    {
        dimensionedScalar p_operateDim("p_operateDim", dimensionSet(1,-1,-2,0,0,0,0),this->incompPref_);  

        if(this->incompPref_ > 0.0) 
        {  
            this->rho_ = p_operateDim*this->psi_; 
        }
        else 
        {
            this->rho_ = this->p_*this->psi_;
        }
    }
    this->rho_inRhoThermo_ = this->rho_;

}
}

// ************************************************************************* //
