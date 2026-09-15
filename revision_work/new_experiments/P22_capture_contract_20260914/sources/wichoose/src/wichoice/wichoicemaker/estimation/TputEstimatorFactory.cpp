/**
 * Implements factory that creates throughput estimators.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <cassert> // assert()
#include <memory>  // std::make_unique

#include "../../../util/conf/WifiStandard.hpp" // enum WifiStandard
#include "../../../util/conf/Network.hpp"      // struct Network
#include "TputEstimatorN.hpp"                  // class TputEstimatorN
#include "TputEstimatorAc.hpp"                 // class TputEstimatorAc
#include "TputEstimatorAd.hpp"                 // class TputEstimatorAd

#include "TputEstimatorFactory.hpp" // class TputEstimatorFactory

/**
 * Creates a vector of throughput estimators for the networks passed in as an argument. Creation is
 * based on the WiFi standard used by each network.
 *
 * @param netMap: a map of networks  to create estimators for.
 * @param tputEstimatorPtrVec: a reference to the vector where the estimators can be found upon
 *                             method return.
 */
void TputEstimatorFactory::createVec(NetworkMap const& netMap,
                                     TputEstimatorPtrVec& tputEstimatorPtrVec) {
  
  tputEstimatorPtrVec.clear(); // ensure clean slate to begin with
  
  for (auto& kvp : netMap) { // for each network
    
    Network const& net = kvp.second;
    
    switch (net.wiStandard) { // switch on the wifi standard

      case WifiStandard::N:
        tputEstimatorPtrVec.push_back(std::make_unique<TputEstimatorN>());
        break;

      case WifiStandard::AC:
        tputEstimatorPtrVec.push_back(std::make_unique<TputEstimatorAc>());
        break;

      case WifiStandard::AD:
        tputEstimatorPtrVec.push_back(std::make_unique<TputEstimatorAd>());
        break;

      default:
        assert(false); // should never happen
    } // switch end
  } // network loop end

}
