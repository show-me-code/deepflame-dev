{
    volScalarField& he = thermo.he();
    #ifdef GPUSolver_
        // start1 = std::clock();
        // t.join();
        // UEqn_GPU.updatePsi(&U[0][0]);
        // K = 0.5*magSqr(U);
        // end1 = std::clock();
        // time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
        // time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
        // time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
        // time_monitor_UinE += double(end1 - start1) / double(CLOCKS_PER_SEC);
        start1 = std::clock();
        // // t.join();
        UEqn_GPU.updatePsi(&U[0][0]);
        K = 0.5*magSqr(U);
        end1 = std::clock();
        time_monitor_UEqn += double(end1 - start1) / double(CLOCKS_PER_SEC);
        time_monitor_UEqn_mtxAssembly += double(end1 - start1) / double(CLOCKS_PER_SEC);
        time_monitor_CPU += double(end1 - start1) / double(CLOCKS_PER_SEC);
    #endif
    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + fvm::div(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      - dpdt
      - fvm::laplacian(turbulence->alphaEff(), he)
      + diffAlphaD
     ==
        fvc::div(hDiffCorrFlux)
    );

    EEqn.relax();

    EEqn.solve();
}
