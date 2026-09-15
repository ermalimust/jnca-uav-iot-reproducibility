/**
 * Defines an abstract class to serve as the basis for concrete network performance database
 * implementations.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NPERF_DB_HPP__
#define NPERF_DB_HPP__

#include <cstddef> // std::size_t
#include <cstdint> // std::uint*_t
#include <utility> // std::pair
#include <map>     // std::map

#include "../../../util/thread/Lockable.hpp" // class Lockable
#include "../../../util/thread/Runnable.hpp" // class Runnable
#include "../../../gps/GpsInfo.hpp"          // struct GpsInfo
#include "../../../dtransfer/tput/Tput.hpp"  // types Tput*
#include "../mobility/MobiKeyFactory.hpp"    // class MobiKeyFactory
#include "../mobility/MobiKey.hpp"           // class MobiKey
#include "../nperfsum/NperfSum.hpp"          // type NperfSumTable
#include "../nperfsum/NperfSumStrat.hpp"     // class NperfSumStrat

#define LOOKAHEAD_DEF 40 // default lookahead window size
#define LOOKAHEAD_MIN 1
#define LOOKAHEAD_MAX 200

// useful type definitions
// we use a copy of the table
typedef std::pair<const bool, const NperfSumTable> NperfSearchRes;

class NperfDb : public Runnable, public Lockable {

public:
  // data-retrieval related methods start
  /**
   * Returns lookahead window size.
   *
   * @return lookahed window size.
   */
  std::size_t getLookahead() const;

  /**
   * Retrieves the network performance summary table associated with the mobility parameters passed as
   * an argument. The retrieval may fail due for lack of data associated with the provided mobility.
   *
   * @param gpsInfo: mobility information for which we want to retrieve a network performance summary.
   * @return a pair made up of a boolean indicating whether the retrieval was successful or not, and a 
   * reference to the requested performance data table, which will only be valid if the boolean is true.
   */
  NperfSearchRes searchNperf(GpsInfo const& gpsInfo);
  // data-retrieval related methods end

  // data-saving related methods start
  /**
   * Adds the provided throughput sample to the database.
   *
   * @param tputSamp: throughput sample to be added to the database.
   * @param tstamp: time at which throughput was collected (secs).
   */
  void addTputSamp(TputSamp const& tputSamp, const uint32_t tstamp);
  
  /**
   * Adds a map of throughput samples, indexed by timestamp, to the database.
   *
   * @param tputSampMap: map with the samples to be added
   */
  void addTputSampMap(TputSampMap const& tputSampMap);
  // data-saving related methods end

  /**
   * Executes a loop that updates mobility history in reaction to changes in gps information.
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;

protected:
  std::size_t nnets;   // number of networks
  std::size_t lookahead; // forecast window length, in seconds

  MobiKeyFactory mobiKeyFactory; // so we can create mobility keys whenever
  MobiKey curMobiKey; // current mobility information

  NperfSumStrat const& nperfSumStrat;
  
  /**
   * Constructor method. To be called from concrete subclasses.
   *
   * @param nperfSumStrat: network performance summarization strategy to use.
   */
  NperfDb(NperfSumStrat const& nperfSumStrat);

  /**
   * Helper method that adds a throughput sample to a given mobility key.
   * Assumes database has been locked prior to call.
   *
   * @param tputSamp: throughput sample to be added to the database.
   * @param offset: how many secs after mobiKey was observed the data was gathered.
   * @param mobiKey: the mobility key the throughput sample should be associated with.
   *
   * @return true if addition was successful, false if offset larger than lookahead.
   */
  virtual bool addTputSampLocked(TputSamp const& tputSamp,
                                 const unsigned offset,
                                 MobiKey const& mobiKey) = 0;
  
  /**
   * Retrieves the network performance summary table associated with the mobility parameters passed as
   * an argument. The retrieval may fail due for lack of data associated with the provided mobility.
   *
   * @param mobiKey: mobility information for which we want to retrieve a network performance summary.
   * @return a pair made up of a boolean indicating whether the retrieval was successful or not, and a
   * reference to the requested performance data table, which will only be valid if the boolean is true.
   */
  virtual NperfSearchRes searchNperf(MobiKey const& mobiKey) = 0;

  /**
   * Resets performance information pertaining to current mobility key.
   * To be called upon switch to new mobility key.
   * Assumes data has been locked prior to call.
   * This is is an abstract method to be implemented by concrete subclasses.
   */
  virtual void resetCurPerfSum() = 0;

  /**
   * Helper method that logs a throughput sample that is being added to the database.
   *
   * @param tputSamp: throughput sample to be logged.
   * @param tstamp: when the data was gathered (secs).
   * @param queued: whether the sample is being added or queued.
  */
  inline void logTputSamp(TputSamp const& tputSamp,
                          const uint32_t tstamp,
                          const bool queued);

private:
  /**
   * Aggregates information needed about each timestamp in mobility history.
   */
  struct MobiHistEntry {
    const MobiKey mobiKey;
    TputSourceSet doneSources; // sources that no longer need this entry

    /**
     * Constructor method.
     *
     * @param the mobility key for the entry.
     */
    MobiHistEntry(MobiKey const& mobiKey);
  };

  typedef std::map<uint32_t, MobiHistEntry> MobiHistMap; // tstamp -> mobi
  MobiHistMap mobiHistMap;  // past mobility keys so we can add forecasts

  typedef std::map<uint32_t, TputSampVec> TputSampQueue; // tstamp -> samp set
  TputSampQueue tputSampQueue; // samples for which mobility is still missing
  
  
  /**
   * Helper method that adds a throughput sample to the database
   * It makes the following assumptions:
   *   1. Database has been locked prior to call;
   *   2. Estimates are saved prior to measurements;
   *   3. Method is called with strictly increasing timestamps.
   *
   * @param tputSamp: throughput sample to be added to the database.
   * @param tstamp: time at which throughput was collected (secs).
   */
  void addTputSampLocked(TputSamp const& tputSamp, const uint32_t tstamp);
};

#endif // NPERF_DB_HPP__
