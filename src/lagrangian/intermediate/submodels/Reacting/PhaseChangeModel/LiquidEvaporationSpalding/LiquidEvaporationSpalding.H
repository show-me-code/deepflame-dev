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

Class
    Foam::LiquidEvaporationSpalding

Description
    Liquid evaporation model
    - uses ideal gas assumption

\*---------------------------------------------------------------------------*/

#ifndef LiquidEvaporationSpalding_H
#define LiquidEvaporationSpalding_H

// #include "PhaseChangeModel.H"
#include "LiquidEvaporationBoil.H"
#include "liquidMixtureProperties.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
/*---------------------------------------------------------------------------*\
                     Class LiquidEvaporationSpalding Declaration
\*---------------------------------------------------------------------------*/

template<class CloudType>
class LiquidEvaporationSpalding
:
    public LiquidEvaporationBoil<CloudType>
{
public:

    //- Runtime type information
    TypeName("liquidEvaporationSpalding");


    // Constructors

        //- Construct from dictionary
        LiquidEvaporationSpalding(const dictionary& dict, CloudType& cloud);

        //- Construct copy
        LiquidEvaporationSpalding(const LiquidEvaporationSpalding<CloudType>& pcm);

        //- Construct and return a clone
        virtual autoPtr<PhaseChangeModel<CloudType>> clone() const
        {
            return autoPtr<PhaseChangeModel<CloudType>>
            (
                new LiquidEvaporationSpalding<CloudType>(*this)
            );
        }


    //- Destructor
    virtual ~LiquidEvaporationSpalding();


    // Member Functions

        //- Update model
        virtual void calculate
        (
            const scalar dt,
            const label celli,
            const scalar Re,
            const scalar Pr,
            const scalar d,
            const scalar nu,
            const scalar T,
            const scalar Ts,
            const scalar pc,
            const scalar Tc,
            const scalarField& X,
            scalarField& dMassPC
        ) const;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "LiquidEvaporationSpalding.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
