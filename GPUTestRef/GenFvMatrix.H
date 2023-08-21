#pragma once

#include "tmp.H"
#include "dimensionedType.H"
#include "volFieldsFwd.H"
#include "surfaceFieldsFwd.H"
#include "typeInfo.H"
#include "runTimeSelectionTables.H"
#include "fvMatrices.H"
#include "fvMesh.H"
#include "turbulentFluidThermoModel.H"
#include "CombustionModel.H"
#include <mpi.h>
#include <algorithm>
#include "PstreamGlobals.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{


// namespace fv
// {

// fvm::ddt
template<class Type>
tmp<fvMatrix<Type>>
EulerDdtSchemeFvmDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<fvMatrix<Type>>
EulerDdtSchemeFvmDdt
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::ddt
template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const volScalarField& rho,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
EulerDdtSchemeFvcDdt
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::ddtCorr
tmp<surfaceScalarField>
EulerDdtSchemeFvcDdtCorr
(
    const volScalarField& rho,
    const volVectorField& U,
    const surfaceScalarField& phi,
    const autoPtr<surfaceVectorField>& Uf
);

tmp<surfaceScalarField>
EulerDdtSchemeFvcDdtPhiCoeff
(
    const volVectorField& U,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiCorr,
    const volScalarField& rho
);

template<class Type>
Foam::tmp<Foam::GeometricField<Type, Foam::fvPatchField, Foam::volMesh>>
UEqn_H
(
    fvMatrix<Type>& UEqn
);

tmp<volScalarField>
rAUConstructor
(
    fvMatrix<vector>& UEqn
);

tmp<surfaceScalarField>
rhorAUfConstructor
(
    const volScalarField& rhorAU,
    const surfaceScalarField& linear_weights
);

tmp<surfaceScalarField>
phiHbyAConstructor
(
    const volScalarField& rho,
    const volVectorField& HbyA,
    const surfaceScalarField& rhorAUf,
    const surfaceScalarField& tddtCorr,
    const surfaceScalarField& linear_weights
);


// fvm::div
template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<fvMatrix<Type>>
gaussConvectionSchemeFvmDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf,
    const word& name
);

// fvc::div
template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const surfaceScalarField& faceFlux,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<GeometricField<Type, fvPatchField, volMesh>>
gaussConvectionSchemeFvcDiv
(
    const GeometricField<Type, fvsPatchField, surfaceMesh>& ssf
);

template<class Type>
tmp
<
    GeometricField
    <
        typename innerProduct<vector, Type>::type, fvPatchField, volMesh
    >
>
gaussDivFvcdiv
(
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvc::grad
template<class Type>
tmp
<
    GeometricField
    <
        typename outerProduct<vector, Type>::type,
        fvPatchField,
        volMesh
    >
>
gaussGradSchemeGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf
);


template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacianUncorrected
(
    const surfaceScalarField& gammaMagSf,
    const surfaceScalarField& deltaCoeffs,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// fvm::laplacian
template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacian
(
    const GeometricField<scalar, fvPatchField, volMesh>& gammaScalarVol,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

template<class Type>
tmp<fvMatrix<Type>>
gaussLaplacianSchemeFvmLaplacian
(
    const GeometricField<scalar, fvsPatchField, surfaceMesh>& gamma,
    const GeometricField<Type, fvPatchField, volMesh>& vf
);

// turbulence->divDevRhoReff(U)
tmp<fvVectorMatrix>
turbulenceModelLinearViscousStressDivDevRhoReff
(
    volVectorField& U,
    compressible::turbulenceModel& turbulence
);

tmp<fvVectorMatrix>
GenMatrix_U(
    const volScalarField& rho,
    volVectorField& U,
    const surfaceScalarField& phi,
    const volScalarField& p, 
    compressible::turbulenceModel& turbulence
);

tmp<fvScalarMatrix>
GenMatrix_Y(
    const volScalarField& rho,
    volScalarField& Yi,
    const surfaceScalarField& phi,
    const surfaceScalarField& phiUc,
    const volScalarField& rhoD,
    const volScalarField& mut,
    const Switch splitting,
    const scalar Sct,
    CombustionModel<basicThermo>& combustion,
    fv::convectionScheme<scalar>& mvConvection
);

tmp<fvScalarMatrix>
GenMatrix_E(
    const volScalarField& rho,
    volScalarField& he,
    const surfaceScalarField& phi,
    const volScalarField& K,
    const volScalarField& dpdt,
    const volScalarField& alphaEff,
    const volScalarField& diffAlphaD,
    const volVectorField& hDiffCorrFlux,
    const surfaceScalarField& linear_weights
);

tmp<fvScalarMatrix>
GenMatrix_p(
    const volScalarField& rho,
    volScalarField& p,
    const surfaceScalarField& phiHbyA,
    const surfaceScalarField& rhorAUf,
    const volScalarField& phi
);


void check_fvmatrix_equal(fvScalarMatrix& a,fvScalarMatrix& b);
void check_fvmatrix_equal(fvVectorMatrix& a,fvVectorMatrix& b);

void check_field_equal(Field<scalar>& a, Field<scalar>& b);


} // End namespace Foam


// ************************************************************************* //