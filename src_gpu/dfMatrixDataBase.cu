#include "dfMatrixDataBase.H"


void constructBoundarySelector(std::vector<int>& patchTypeSelector, const std::string& patchTypeStr,
    const int patchSize)
{
    boundaryConditions patchCondition;
    std::vector<int> tmpSelector;
    static std::map<std::string, boundaryConditions> BCMap = {
        {"zeroGradient", zeroGradient},
        {"fixedValue", fixedValue},
        {"empty", empty},
        {"coupled", coupled}
    };
    auto iter = BCMap.find(patchTypeStr);
    if (iter != BCMap.end()) {
        patchCondition = iter->second;
    } else {
        throw std::runtime_error("Unknown boundary condition: " + patchTypeStr);
    }
    // zeroGradient labeled as 0, fixedValue labeled as 1, coupled labeled as 2
    switch (patchCondition){
        case zeroGradient:
        {
            tmpSelector.resize(patchSize, 0);
            patchTypeSelector.insert(patchTypeSelector.end(), tmpSelector.begin(), tmpSelector.end());
            break;
        }
        case fixedValue:
        {
            tmpSelector.resize(patchSize, 1);
            patchTypeSelector.insert(patchTypeSelector.end(), tmpSelector.begin(), tmpSelector.end());
            break;
        }
        case empty:
        {
            tmpSelector.resize(patchSize, 2);
            patchTypeSelector.insert(patchTypeSelector.end(), tmpSelector.begin(), tmpSelector.end());
            break;
        }
        case coupled:
        {
            tmpSelector.resize(patchSize, 3);
            patchTypeSelector.insert(patchTypeSelector.end(), tmpSelector.begin(), tmpSelector.end());
            break;
        }
    }
}
