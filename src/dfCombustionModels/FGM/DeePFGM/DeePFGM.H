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

Class
    Foam::combustionModels::flareFGM

Description
    Partially stirred reactor turbulent combustion model.

    This model calculates a finite rate, based on both turbulence and chemistry
    time scales.  Depending on mesh resolution, the Cmix parameter can be used
    to scale the turbulence mixing time scale.

SourceFiles
    flareFGM.C

\*---------------------------------------------------------------------------*/

#ifndef DeePFGM_H
#define DeePFGM_H

#include "baseFGM.H"
#include "tableSolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace combustionModels
{

/*---------------------------------------------------------------------------*\
                            Class flareFGM Declaration
\*---------------------------------------------------------------------------*/

template<class ReactionThermo>
class DeePFGM
:
    public baseFGM<ReactionThermo>,
    public tableSolver
{

public:

    //- Runtime type information
    TypeName("DeePFGM");


    // Constructors

        //- Construct from components
        DeePFGM
        (
            const word& modelType,
            ReactionThermo& thermo,
            const compressibleTurbulenceModel& turb,
            const word& combustionProperties
        );

        //- Disallow default bitwise copy construction
        DeePFGM(const DeePFGM&);


    //- Destructor
    virtual ~DeePFGM();


    // Member Operators

        //- Correct combustion rate
        virtual void correct();

        //- retrieval data from table
        // virtual void retrieval();

        virtual void retrieval(PyObject* module,PyObject* func);
        virtual int prediction(
        double z_s[], double c_s[], double gz_s[], double gc_s[],
        double gcz_s[],  int phinum, int dimension,double* result,int size,PyObject* module,PyObject* func
    );
        virtual PyObject* initialize_module();
        virtual PyObject* initialize_function(PyObject* module);

        //- Disallow default bitwise assignment
        void operator=(const DeePFGM&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "DeePFGM.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
