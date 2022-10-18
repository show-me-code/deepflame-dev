Utility
=================

DeepFlame uses *yaml* reaction mechanisms, which are compatible with Cantera. The following command lines can be used to convert *chemkin* mechanisms into *yaml* format. 


.. code-block:: bash

    conda create --name ct-env --channel cantera cantera ipython matplotlib jupyter
    conda activate ct-env
    ck2yaml --input=chem.inp --thermo=therm.dat --transport=tran.dat

.. Note:: If there is an **ImportError** regarding *libmkl_rt.so.2*, go to your ct-env folder to check the existence of *libmkl_rt.so.2* and *libmkl_rt.so.1*, and replace *libmkl_rt.so.1* with *libmkl_rt.so.2*.

More detailed instruction of converting mechanisms can be found on `Cantera official website <https://cantera.org/tutorials/ck2yaml-tutorial.html>`_. 