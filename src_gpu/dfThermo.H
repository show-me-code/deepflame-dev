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

    double *d_mole_fraction, *d_mean_mole_weight;
    double *d_boundary_mole_fraction, *d_boundary_mean_mole_weight;

    // intermediate variables
    std::vector<double> T_poly;

    double *d_T_poly, *d_boundary_T_poly;
    double *d_species_viscosities, *d_boundary_species_viscosities;
    double *d_species_thermal_conductivities, *d_boundary_species_thermal_conductivities;

    // constructor
    dfThermo(dfMatrixDataBase &dataBase)
        : dataBase_(dataBase) {};

    // destructor
    ~dfThermo(){};

    // public member functions
    void setConstantValue(std::string mechanism_file, int num_cells, int num_species);
    void setConstantFields(const std::vector<int> patch_type);
    void initNonConstantFields(double *h_T, double *h_he, double *h_psi, double *h_alpha, double *h_mu,
            double *h_boundary_T, double *h_boundary_he, double *h_boundary_psi, double *h_boundary_alpha, double *h_boundary_mu);

    // set mass fraction
    void setMassFraction(const double *d_y, const double *d_boundary_y);

    // *** GPU functions ***
    void calculateTPolyGPU(int num_thread, const double *T, double *T_poly, int offset = 0);
    void calculatePsiGPU(int num_thread, const double *T, double *psi, int offset = 0);
    void calculateRhoGPU(int num_thread, const double *p, const double *psi, double *rho, int offset = 0);
    void calculateViscosityGPU(int num_thread, int num_total, const double *T, const double *mole_fraction, 
            const double *T_poly, double *species_viscosities, double *viscosity, int offset = 0);
    void calculateThermoConductivityGPU(int num_thread, int num_total, const double *T, const double *T_poly,
            const double *d_y, const double *mole_fraction, double *species_thermal_conductivities,
            double *thermal_conductivity, int offset = 0);
    void calculateEnthalpyGPU(int num_thread, const double *T, double *enthalpy, const double *d_mass_fraction, int offset = 0);
    void calculateTemperatureGPU(int num_thread, int num_total, const double *T_init, const double *target_h, 
            double *T, const double *d_mass_fraction, int offset = 0,
            double atol = 1e-7, double rtol = 1e-7, int max_iter = 20);
    
    void calculateEnergyGradient(int num_thread, int num_cells, int num_species, 
            int num_boundary_surfaces, int bou_offset, int gradient_offset, const int *face2Cells, 
            const double *T, const double *p, const double *y, const double *boundary_delta_coeffs,
            const double *boundary_p, const double *boundary_y, double *boundary_thermo_gradient);
    
    void compareThermoConductivity(const double *d_thermal_conductivity, const double *thermal_conductivity,
            bool printFlag);
    void compareViscosity(const double *d_viscosity, const double *viscosity, bool printFlag);
    void compareHe(const double *he);

    void sync();

    // outer API
    void updateEnergy();
    void correctThermo();

    // getter functions
};