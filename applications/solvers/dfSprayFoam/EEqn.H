{
    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      - fvm::laplacian(turbulence->alphaEff(), he)
      + diffAlphaD
     ==
        rho*(U&g)
      + parcels.Sh(he)
      + fvc::div(hDiffCorrFlux)
      //+ radiation->Sh(thermo, he)
      //+ fvOptions(rho, he)
    );

    EEqn.relax();

    //fvOptions.constrain(EEqn);

    EEqn.solve();

    //fvOptions.correct(he);

    //radiation->correct();
}
