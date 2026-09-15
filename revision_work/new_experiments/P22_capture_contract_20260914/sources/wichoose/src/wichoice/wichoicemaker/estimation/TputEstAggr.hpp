/**
 * Defines class to periodically (1 Hz) estimate network throughput and save it on a database.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef TPUT_EST_AGGR_HPP__
#define TPUT_EST_AGGR_HPP__

#include "../../../util/thread/Runnable.hpp" // class Runnable
#include "../../../util/conf/Network.hpp"    // type NetworkMap
#include "../database/NperfDb.hpp"           // class NperfDb

class TputEstAggr : public Runnable {

public:
  /**
   * Constructor.
   *
   * @param nperfDb: the database object to save throughput estimates to.
   * @param printEst: whether the throughput estimates should be printed to the standard output.
   */
  TputEstAggr(NperfDb& nperfDb, const bool printEst);

  /**
   * Executes a loop that periodically estimates network performance and saves it to the associated
   * database.
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;

private:
  NperfDb& nperfDb;
  NetworkMap netMap; // map with all networks of interest
  const bool printEst;
};

#endif // TPUT_EST_AGGR_HPP__
