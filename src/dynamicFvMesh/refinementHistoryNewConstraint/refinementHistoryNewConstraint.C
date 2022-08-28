/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2015-2016 OpenFOAM Foundation
     \\/     M anipulation  | Copyright (C) 2018 OpenCFD Ltd.
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

#include "refinementHistoryNewConstraint.H"
#include "addToRunTimeSelectionTable.H"
#include "syncTools.H"
#include "refinementHistoryNew.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace decompositionConstraints
{
    defineTypeName(refinementHistoryNewConstraint);

    addToRunTimeSelectionTable
    (
        decompositionConstraint,
        refinementHistoryNewConstraint,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::decompositionConstraints::refinementHistoryNewConstraint::refinementHistoryNewConstraint
(
    const dictionary& dict,
    const word& modelType
)
:
    decompositionConstraint(dict, typeName)
{
    if (decompositionConstraint::debug)
    {
        Info<< type()
            << " : setting constraints to refinement history New" << endl;
    }
}


Foam::decompositionConstraints::refinementHistoryNewConstraint::refinementHistoryNewConstraint()
:
    decompositionConstraint(dictionary(), typeName)
{
    if (decompositionConstraint::debug)
    {
        Info<< type()
            << " : setting constraints to refinement history New" << endl;
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::decompositionConstraints::refinementHistoryNewConstraint::add
(
    const polyMesh& mesh,
    boolList& blockedFace,
    PtrList<labelList>& specifiedProcessorFaces,
    labelList& specifiedProcessor,
    List<labelPair>& explicitConnections
) const
{
    // The refinement history type
    typedef ::Foam::refinementHistoryNew HistoryType;
    // typedef Foam::decompositionConstraints::refinementHistoryNew HistoryType;

    // Local storage if read from file
    autoPtr<const HistoryType> readFromFile;

    const HistoryType* historyPtr = nullptr;

    if (mesh.foundObject<HistoryType>("refinementHistoryNew"))
    {
        if (decompositionConstraint::debug)
        {
            Info<< type() << " : found refinementHistoryNew" << endl;
        }
        historyPtr = &mesh.lookupObject<HistoryType>("refinementHistoryNew");
    }
    else
    {
        if (decompositionConstraint::debug)
        {
            Info<< type() << " : reading refinementHistoryNew from time "
                << mesh.facesInstance() << endl;
        }

        readFromFile.reset
        (
            new HistoryType
            (
                IOobject
                (
                    "refinementHistoryNew",
                    mesh.facesInstance(),
                    polyMesh::meshSubDir,
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE
                ),
                mesh.nCells()
            )
        );

        // historyPtr = readFromFile.get();  // get(), not release()
    }

    // const auto& history = *historyPtr;
    const auto& history =
    (
        readFromFile.valid()
       ? readFromFile()
       : *historyPtr
    );

    if (history.active())
    {
        // refinementHistoryNew itself implements decompositionConstraint
        history.add
        (
            blockedFace,
            specifiedProcessorFaces,
            specifiedProcessor,
            explicitConnections
        );
    }
}


void Foam::decompositionConstraints::refinementHistoryNewConstraint::apply
(
    const polyMesh& mesh,
    const boolList& blockedFace,
    const PtrList<labelList>& specifiedProcessorFaces,
    const labelList& specifiedProcessor,
    const List<labelPair>& explicitConnections,
    labelList& decomposition
) const
{
    // The refinement history type
    typedef ::Foam::refinementHistoryNew HistoryType;

    // Local storage if read from file
    autoPtr<const HistoryType> readFromFile;

    const HistoryType* historyPtr = nullptr;

    if(mesh.foundObject<HistoryType>("refinementHistoryNew"))
    {
        historyPtr = &mesh.lookupObject<HistoryType>("refinementHistoryNew");
    }
    else
    {
        readFromFile.reset
        (
            new HistoryType
            (
                IOobject
                (
                    "refinementHistoryNew",
                    mesh.facesInstance(),
                    polyMesh::meshSubDir,
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE
                ),
                mesh.nCells()
            )
        );

        // historyPtr = readFromFile.get();  // get(), not release()
    }

    // const auto& history = *historyPtr;
    const auto& history =
    (
        readFromFile.valid()
       ? readFromFile()
       : *historyPtr
    );

    if (history.active())
    {
        // refinementHistoryNew itself implements decompositionConstraint
        history.apply
        (
            blockedFace,
            specifiedProcessorFaces,
            specifiedProcessor,
            explicitConnections,
            decomposition
        );
    }
}


// ************************************************************************* //