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

    start1 = std::clock();
    fvScalarMatrix rhoEqn
    (
        fvm::ddt(rho)
      + fvc::div(phi)
      ==
        parcels.Srho(rho)
    );
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);

    start1 = std::clock();
    rhoEqn.solve();
    end1 = std::clock();
    time_monitor_rhoEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
    time_monitor_rhoEqn_solve += double(end1 - start1) / double(CLOCKS_PER_SEC);
    
// ************************************************************************* //
