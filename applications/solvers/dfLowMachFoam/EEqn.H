{
    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn
        (

            fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
        +   fvc::ddt(rho, K) + fvc::div(phi, K)
        -   dpdt
            turbName == "laminar"
            ?
            (
            -   fvm::laplacian(turbulence->alpha(), he)
            +   diffAlphaD
            ==  
                fvc::div(hDiffCorrFlux)
            )
            :
            (
            -   fvm::laplacian(turbulence->alphaEff(), he)
            )
        );

        EEqn.relax();

        EEqn.solve("ha");
}
