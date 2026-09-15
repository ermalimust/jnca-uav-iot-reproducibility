/**
 * Defines a class that maintains an in-memory database of throughput information indexed by mobility data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NPERF_MEM_DB_HPP__
#define NPERF_MEM_DB_HPP__

#include <cstdint> // std::uint*_t
#include <cstddef> // std::size_t
#include <map>     // std::map

#include "../../../dtransfer/tput/Tput.hpp" // types Tput*
#include "../mobility/MobiKey.hpp"          // class MobiKey
#include "../nperfsum/NperfSum.hpp"          // type NperfSumTable
#include "../nperfsum/NperfSumStrat.hpp"    // class NperfSumStrat
#include "NperfDb.hpp"                      // class NperfDb

// useful type definitions
class NperfMemDb : public NperfDb {
  
public:
  /**
   * Constructor method.
   *
   * @param nperfSumStrat: network performance summarization strategy to use.
   */
  NperfMemDb(NperfSumStrat const& nperfSumStrat);
  
  /**
   * Load the database contents from disk.
   */
  void loadFromDisk();

  /**
   * Save the entire database onto disk for later use.
   */
  void saveToDisk();

protected:
  /**
   * Retrieves the network performance summary table associated with the mobility parameters passed as
   * an argument. The retrieval may fail due for lack of data associated with the provided mobility.
   * Overrides abstract method from superclass.
   *
   * @param mobiKey: mobility information for which we want to retrieve a network performance summary.
   * @return a pair made up of a boolean indicating whether the retrieval was successful or not, and a
   * reference to the requested performance data table, which will only be valid if the boolean is true.
   */
  NperfSearchRes searchNperf(MobiKey const& mobiKey) override;
  
  /**
   * Helper method that adds a throughput sample to a given mobility key.
   * Assumes database has been locked prior to call.
   * Overrides abstract method from superclass.
   *
   * @param tputSamp: throughput sample to be added to the database.
   * @param offset: how many secs after mobiKey was observed the data was gathered.
   * @param mobiKey: the mobility key the throughput sample should be associated with.
   *
   * @return true if addition was successful, false if offset larger than lookahead.
   */
  bool addTputSampLocked(TputSamp const& tputSamp,
                         const unsigned offset,
                         MobiKey const& mobiKey) override;
  
  /**
   * Resets performance information pertaining to current mobility key.
   * To be called upon switch to new mobility key.
   * Assumes data has been locked prior to call.
   * Overrides abstract method from superclass.
   */
  virtual void resetCurPerfSum() override;
    
private:
  typedef std::map<MobiKey, NperfSumTable> NperfSumTableMap; // mobi -> table
  
  NperfSumTableMap dataMap; // actual network performance data in database
  std::string dbFname; // filename used to persist/load data to/from disk
  
  NperfSumTable curNperfSumTable; // perf table for current mobility key
  
  /**
   * Helper method that adds a throughput sample to a performance summary table.
   * Assumes database has been locked prior to call.
   *
   * @param tputSamp: throughput sample to be added to the database.
   * @param offset: offset to index the table, representing how far into the future (s) data was gathered.
   * @param nperfSumTable: the table onto which the throughput sample is to be added.
   *
   * @return true if addition was successful, false if offset larger than lookahead.
   */
  bool addTputSampLocked(TputSamp const& tputSamp,
                         const unsigned offset,
                         NperfSumTable& nperfSumTable);

  /**
   * If the provided mobility key already exists in the historical data map, the table currently associated with 
   * the key is returned. If it does not, it is added, in association with a blank performance table. Said table is
   * then returned to the caller.
   *
   * @param mobiKey: the mobility key to index the data map with.
   * @return reference to the performance table associated with mobiKey.
   */
  NperfSumTable& getHistNperfSumTable(MobiKey const& mobiKey);
  
};

#endif // NPERF_MEM_DB_HPP__
