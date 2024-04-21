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

#include "PatchFuncInjection.H"
#include "TimeFunction1.H"
#include "distributionModel.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class CloudType>
Foam::PatchFuncInjection<CloudType>::PatchFuncInjection
(
    const dictionary& dict,
    CloudType& owner,
    const word& modelName
)
:
    InjectionModel<CloudType>(dict, owner, modelName, typeName),
    patchInjectionBase(owner.mesh(), this->coeffDict().lookup("patchName")),
    duration_(readScalar(this->coeffDict().lookup("duration"))),
    parcelsPerSecond_
    (
        readScalar(this->coeffDict().lookup("parcelsPerSecond"))
    ),
    U0_(this->coeffDict().lookup("U0")),
    flow_direction_(this->coeffDict().lookup("flow_direction")),
    U_func_a_(readScalar(this->coeffDict().lookup("U_func_a"))),
    U_func_b_(readScalar(this->coeffDict().lookup("U_func_b"))),
    U_func_c_(readScalar(this->coeffDict().lookup("U_func_c"))),
    U_func_d_(readScalar(this->coeffDict().lookup("U_func_d"))),
    flowRateProfile_
    (
        TimeFunction1<scalar>
        (
            owner.db().time(),
            "flowRateProfile",
            this->coeffDict()
        )
    ),
    sizeDistribution_
    (
        distributionModel::New
        (
            this->coeffDict().subDict("sizeDistribution"),
            owner.rndGen()
        )
    )
{
    duration_ = owner.db().time().userTimeToTime(duration_);

    patchInjectionBase::updateMesh(owner.mesh());

    // Set total volume/mass to inject
    this->volumeTotal_ = flowRateProfile_.integrate(0.0, duration_);
}


template<class CloudType>
Foam::PatchFuncInjection<CloudType>::PatchFuncInjection
(
    const PatchFuncInjection<CloudType>& im
)
:
    InjectionModel<CloudType>(im),
    patchInjectionBase(im),
    duration_(im.duration_),
    parcelsPerSecond_(im.parcelsPerSecond_),
    U0_(im.U0_),
    flowRateProfile_(im.flowRateProfile_),
    sizeDistribution_(im.sizeDistribution_().clone().ptr())
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class CloudType>
Foam::PatchFuncInjection<CloudType>::~PatchFuncInjection()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class CloudType>
void Foam::PatchFuncInjection<CloudType>::updateMesh()
{
    patchInjectionBase::updateMesh(this->owner().mesh());
}


template<class CloudType>
Foam::scalar Foam::PatchFuncInjection<CloudType>::timeEnd() const
{
    return this->SOI_ + duration_;
}


template<class CloudType>
Foam::label Foam::PatchFuncInjection<CloudType>::parcelsToInject
(
    const scalar time0,
    const scalar time1
)
{
    if ((time0 >= 0.0) && (time0 < duration_))
    {
        scalar nParcels = (time1 - time0)*parcelsPerSecond_;

        Random& rnd = this->owner().rndGen();

        label nParcelsToInject = floor(nParcels);

        // Inject an additional parcel with a probability based on the
        // remainder after the floor function
        if
        (
            nParcelsToInject > 0
         && (nParcels - scalar(nParcelsToInject) > rnd.globalScalar01())
        )
        {
            ++nParcelsToInject;
        }

        return nParcelsToInject;
    }
    else
    {
        return 0;
    }
}


template<class CloudType>
Foam::scalar Foam::PatchFuncInjection<CloudType>::volumeToInject
(
    const scalar time0,
    const scalar time1
)
{
    if ((time0 >= 0.0) && (time0 < duration_))
    {
        return flowRateProfile_.integrate(time0, time1);
    }
    else
    {
        return 0.0;
    }
}


template<class CloudType>
void Foam::PatchFuncInjection<CloudType>::setPositionAndCell
(
    const label,
    const label,
    const scalar,
    vector& position,
    label& cellOwner,
    label& tetFacei,
    label& tetPti
)
{
    patchInjectionBase::setPositionAndCell
    (
        this->owner().mesh(),
        this->owner().rndGen(),
        position,
        cellOwner,
        tetFacei,
        tetPti
    );
}


template<class CloudType>
void Foam::PatchFuncInjection<CloudType>::setProperties
(
    const label,
    const label,
    const scalar,
    typename CloudType::parcelType& parcel
)
{
    // set particle velocity
    scalar x = parcel.position()[0];
    scalar y = parcel.position()[1];
    scalar z = parcel.position()[2];
    std::cout << "get functional parcel position: " << x << "  " << y << "  " << z <<std::endl;

    vector U_func = this->U0_;
    if (flow_direction_ == "x")
    {
        scalar r = std::sqrt(z*z+y*y);
        U_func = vector(U_func_a_*exp(U_func_b_*r) + U_func_c_*exp(U_func_d_*r), 0, 0);
    }
    else if (flow_direction_ == "y")
    {
        scalar r = std::sqrt(x*x+z*z);
        U_func = vector(0, U_func_a_*exp(U_func_b_*r) + U_func_c_*exp(U_func_d_*r), 0);
    }
    else if (flow_direction_ == "z")
    {
        scalar r = std::sqrt(x*x+y*y);
        U_func = vector(0, 0, U_func_a_*exp(U_func_b_*r) + U_func_c_*exp(U_func_d_*r));
    }

    std::cout << "set functional parcel velocity: " << U_func.x() << "  "  << U_func.y()<< "  "  << U_func.z() <<std::endl;
    parcel.U() = U_func;

    // set particle diameter
    parcel.d() = sizeDistribution_->sample();
}


template<class CloudType>
bool Foam::PatchFuncInjection<CloudType>::fullyDescribed() const
{
    return false;
}


template<class CloudType>
bool Foam::PatchFuncInjection<CloudType>::validInjection(const label)
{
    return true;
}


// ************************************************************************* //
