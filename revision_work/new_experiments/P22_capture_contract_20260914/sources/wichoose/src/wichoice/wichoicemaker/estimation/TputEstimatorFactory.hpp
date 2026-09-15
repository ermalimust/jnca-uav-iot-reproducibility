/**
 * Defines factory to create throughput estimator objects.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef TPUT_ESTIMATOR_FACTORY_HPP__
#define TPUT_ESTIMATOR_FACTORY_HPP__

#include "../../../util/conf/Network.hpp" // type NetworkMap
#include "TputEstimator.hpp"              // type TputEstimatorPtrVec

class TputEstimatorFactory {

public:
  /**
   * Creates a vector of throughput estimators for the networks passed in as an argument. Creation is
   * based on the WiFi standard used by each network.
   *
   * @param netMap: a map of networks  to create estimators for.
   * @param tputEstimatorPtrVec: a reference to the vector where the estimators can be found upon
   *                             method return.
   */
  static void createVec(NetworkMap const& netMap,
                        TputEstimatorPtrVec& tputEstimatorPtrVec);
};

#endif // TPUT_ESTIMATOR_FACTORY_HPP__
