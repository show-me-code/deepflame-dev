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
    centralCourantNo

Description
    Calculates the mean and maximum wave speed based Courant Numbers.

\*---------------------------------------------------------------------------*/
surfaceScalarField rho_pos(interpolate(rho, pos));
surfaceScalarField rho_neg(interpolate(rho, neg));

surfaceVectorField rhoU_pos(interpolate(rhoU, pos, U.name()));
surfaceVectorField rhoU_neg(interpolate(rhoU, neg, U.name()));

volScalarField rPsi("rPsi", 1.0/psi);
volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));
surfaceScalarField rPsi_pos(interpolate(rPsi, pos, T.name()));
surfaceScalarField rPsi_neg(interpolate(rPsi, neg, T.name()));

// surfaceVectorField U_pos("U_pos", rhoU_pos/rho_pos);
// surfaceVectorField U_neg("U_neg", rhoU_neg/rho_neg);

surfaceVectorField U_pos(interpolate(U, pos, U.name()));
surfaceVectorField U_neg(interpolate(U, neg, U.name()));

surfaceScalarField phiv_pos("phiv_pos", U_pos & mesh.Sf());
surfaceScalarField phiv_neg("phiv_neg", U_neg & mesh.Sf());

// Make fluxes relative to mesh-motion
if (mesh.moving())
{
  phiv_pos -= mesh.phi();
  phiv_neg -= mesh.phi();
}

surfaceScalarField cSf_pos
(
  "cSf_pos",
  interpolate(c, pos, T.name())*mesh.magSf()
);
surfaceScalarField cSf_neg
(
  "cSf_neg",
  interpolate(c, neg, T.name())*mesh.magSf()
);

surfaceScalarField ap
(
  "ap",
  max(max(phiv_pos + cSf_pos, phiv_neg + cSf_neg), v_zero)
);
surfaceScalarField am
(
  "am",
  min(min(phiv_pos - cSf_pos, phiv_neg - cSf_neg), v_zero)
);

surfaceScalarField a_pos("a_pos", ap/(ap - am));

surfaceScalarField amaxSf("amaxSf", max(mag(am), mag(ap)));

surfaceScalarField aSf("aSf", am*a_pos);

word fluxScheme("Kurganov");
mesh.schemesDict().readIfPresent("fluxScheme", fluxScheme);
if (fluxScheme == "Tadmor")
{
  aSf = -0.5*amaxSf;
  a_pos = 0.5;
}

surfaceScalarField a_neg("a_neg", 1.0 - a_pos);

phiv_pos *= a_pos;
phiv_neg *= a_neg;

surfaceScalarField aphiv_pos("aphiv_pos", phiv_pos - aSf);
surfaceScalarField aphiv_neg("aphiv_neg", phiv_neg + aSf);

// Reuse amaxSf for the maximum positive and negative fluxes
// estimated by the central scheme
amaxSf = max(mag(aphiv_pos), mag(aphiv_neg));

if (mesh.nInternalFaces())
{
    scalarField sumAmaxSf(fvc::surfaceSum(amaxSf)().primitiveField());

    CoNum = 0.5*gMax(sumAmaxSf/mesh.V().field())*runTime.deltaTValue();

    meanCoNum =
        0.5*(gSum(sumAmaxSf)/gSum(mesh.V().field()))*runTime.deltaTValue();
}

Info<< "Mean and max Courant Numbers = "
    << meanCoNum << " " << CoNum << endl;

// ************************************************************************* //
