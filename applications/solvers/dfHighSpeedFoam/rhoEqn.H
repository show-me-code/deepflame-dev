if ((ddtSchemes == "RK2SSP") || (ddtSchemes == "RK3SSP"))
{
    rho_rhs = -fvc::div(rhoPhi);

    rho =   rkcoe1[nrk]*rho_save
          + rkcoe2[nrk]*rho 
          + rkcoe3[nrk]*rho_rhs*runTime.deltaT();

    Info <<"in rk"<< nrk+1 << " finish calculate rho" << nl << endl;
}
else
{
    // use to analysis 
    volScalarField::Internal massSource(0.0*Sm); 
    forAll(parcels.rhoTrans(),i)
    {
        massSource += parcels.rhoTrans(i)/runTime.deltaT()/mesh.V();
    }
    Sm = massSource;

    solve
    (
        fvm::ddt(rho)
      + fvc::div(rhoPhi)
      ==
        parcels.Srho(rho)
    );
}


