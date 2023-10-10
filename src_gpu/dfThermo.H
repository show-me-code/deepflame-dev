#pragma once

#include "dfMatrixDataBase.H"

void init_const_coeff_ptr(std::vector<std::vector<double>>& nasa_coeffs, std::vector<std::vector<double>>& viscosity_coeffs,
        std::vector<std::vector<double>>& thermal_conductivity_coeffs, std::vector<std::vector<double>>& binary_diffusion_coeffs,
        std::vector<double>& molecular_weights);

class dfThermo
{
    dfMatrixDataBase &dataBase_;

    // private data members
    std::string mechanism_file;
    std::string thermo_coeff_file;

    // private member functions
    void readCoeffsBinary(FILE* fp, int dimension, std::vector<std::vector<double>>& coeffs);
    void initCoeffsfromBinaryFile(FILE* fp);

public:
    // cuda resource
    cudaStream_t stream;

    // public data members
    int num_species;
    int num_cells;

    // thermo coeffs
    std::vector<std::vector<double>> nasa_coeffs;
    std::vector<std::vector<double>> viscosity_coeffs;
    std::vector<std::vector<double>> thermal_conductivity_coeffs;
    std::vector<std::vector<double>> binary_diffusion_coeffs;
    std::vector<double> molecular_weights;

    // species info
    std::vector<double> mass_fraction;
    std::vector<double> mole_fraction;
    double meanMolecularWeight;

    double *d_mass_fraction, *d_mole_fraction, *d_mean_mole_weight;

    // intermediate variables
    std::vector<double> T_poly;

    double *d_T_poly;
    double *d_species_viscosities, *d_species_thermal_conductivities;

    // constructor
    dfThermo(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {};

    // destructor
    ~dfThermo(){};

    // public member functions
    void setConstantValue(std::string mechanism_file, int num_cells, int num_species);
    void setConstantFields(const std::vector<int> patch_type);
    void initNonConstantFields(double *h_T, double *h_he, double *h_boundary_T, double *h_boundary_he);

    // set mass fraction
    void setMassFraction(const double *d_mass_fraction);

    // *** GPU functions ***
    void calculateTPolyGPU(const double *T);
    void calculatePsiGPU(const double *T, double *psi);
    void calculateRhoGPU(const double *p, const double *psi, double *rho);
    void calculateViscosityGPU(const double *T, double *viscosity);
    void calculateThermoConductivityGPU(const double *T, const double *d_mass_fraction, double *thermal_conductivity);
    void calculateEnthalpyGPU(int num, const double *T, double *enthalpy, const double *d_mass_fraction, int offset = 0);
    void calculateTemperatureGPU(const double *T_init, const double *target_h, double *T, const double *d_mass_fraction, 
            double atol = 1e-7, double rtol = 1e-7, int max_iter = 20);
    
    void compareThermoConductivity(const double *d_thermal_conductivity, const double *thermal_conductivity,
            bool printFlag);
    void compareViscosity(const double *d_viscosity, const double *viscosity, bool printFlag);
    void compareHe(const double *he);

    void sync();

    // outer API
    void updateEnergy();

    // getter functions
};