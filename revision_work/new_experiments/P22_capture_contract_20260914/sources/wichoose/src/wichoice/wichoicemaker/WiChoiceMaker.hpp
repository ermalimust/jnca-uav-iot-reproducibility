/**
 * Defines a class to periodically choose the best network to use as a function of present mobility and past
 * mobility-indexed network performance.
 * It is an abstract class. Concrete subclasses must decide whether to use estimated or measurement
 * data in their computations.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_MAKER_HPP__
#define WI_CHOICE_MAKER_HPP__

#include <cstddef> // std::size_t
#include <cstdint> // uint_*_t
#include <vector>  // std::vector
#include <chrono>  // std::chrono
#include <string>  // std::string

#include "../../util/thread/Runnable.hpp" // class Runnable
#include "../../util/conf/Network.hpp"    // type NetworkMap
#include "../shm/WiChoiceWriter.hpp"      // class WiChoiceWriter
#include "database/NperfDb.hpp"           // class NetPerfDb
#include "nperfsum/NperfSum.hpp"         // struct NperfSums, type NperfSumTable
#include "nperfsum/NperfSumStrat.hpp"     // class NperfSumStrat

#define CHOICE_DELAY_DEF 400  // default delay before interface choice (ms)
#define CHOICE_DELAY_MIN 0
#define CHOICE_DELAY_MAX 700

class WiChoiceMaker : public Runnable {

public:
  /**
   * Destructor. Needs to be made virtual on abstract classes to ensure proper destruction.
   */
  virtual ~WiChoiceMaker();

  /**
   * Executes a loop that reacts to changes in mobility conditions by computing the interface to use in order
   * to maxime throughput under those new conditions, and writes that choice to the dedicated shared
   * memory region.
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;
  
protected:
  NperfSumStrat const& nperfSumStrat;

  /**
   * Constructor method. Protected because this is an abstract class meant to be overwritten.
   *
   * @param nperfDb: historical network performance database to use when picking an interface.
   * @param nperfSumStrat: network performance summarization strategy to use.
   */
  WiChoiceMaker(NperfDb& nperfDb, 
                NperfSumStrat const& nperfSumStrat,
                std::string const& printTag);

  /**
   * Compute and return the expected throughput from the provided performance summaries.
   * The summaries include both measured and estimated data. This method is abstract. Concrete
   * subclasses need to choose between the two types of data.
   *
   * @param nperfSums: the performance summaries to compute a mean from.
   * @return the mean throughput from the summaries, either estimated or measured.
   */
  virtual uint64_t getExpTput(NperfSums const& nperfSums) = 0;
  
  /**
   * Returns the object that should be used to write new wifi interface choices to shared memory.
   * This is an abstract method that needs to be implemented by concrete subclasses.
   *
   * @return object to use to write new wifi interface choices to shared memory.
   */
  virtual WiChoiceWriter& getWiChoiceWriter() = 0;

private:
  // useful type definitions
  typedef std::vector<uint64_t> U64Vector;
  typedef std::vector<U64Vector> U64Table;
  
  NperfDb& nperfDb;           // to read network performance info
  const std::string printTag; // to distinguish different choice maker variants
  
  NetworkMap netMap;
  std::size_t nnets; // simply for efficiency

  std::size_t lookahead;    // forecast window length, in seconds
  unsigned netSwitchTime; // length of time offline during network switch

  std::chrono::milliseconds choiceDelay; // wait after timestamp change
  
  U64Table tdata; // transferable data table - used by findMaxDataIface
                  // it's a field to eliminate the need for constant recreation
  
  /**
   * Determine and return the network that we should pick at the current time in order to maximize the total
   * amount of data that can be transferred over the entire lookahead window, according to past performance.
   * A dynamic-programming algorithm is used for this purpose.
   *
   * @param curNetId: the currently selected network.
   * @param nperfSumTable: historical performance over the lookahead window for each network.
   * @return the id of the network that maximizes the total amount of transferable data.
   */
  unsigned findMaxDataNet(const unsigned curNetId,
                          NperfSumTable const& nperfSumTable);

};

#endif // WI_CHOICE_MAKER_HPP__
