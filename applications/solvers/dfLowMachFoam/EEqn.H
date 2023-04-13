{
    volScalarField& he = thermo.he();

    if ( turbName == "laminar")
    {
        fvScalarMatrix EEqn
        (
            fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
        +   fvc::ddt(rho, K) + fvc::div(phi, K)
        -   dpdt
        -   fvm::laplacian(turbulence->alphaEff(), he)
        +   diffAlphaD
        ==
            fvc::div(hDiffCorrFlux)
        );

        EEqn.relax();

        EEqn.solve();
    }
    else
    {
        fvScalarMatrix EEqn
        (
            fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
        +   fvc::ddt(rho, K) + fvc::div(phi, K)
        -   dpdt
        -   fvm::laplacian(turbulence->alphaEff(), he)
        );

        EEqn.relax();

        EEqn.solve();
    }


}
