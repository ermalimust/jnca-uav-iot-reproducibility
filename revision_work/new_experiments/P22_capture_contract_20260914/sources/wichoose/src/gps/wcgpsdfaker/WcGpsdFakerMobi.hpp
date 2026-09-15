/**
 * Defines class for a GPS daemon that uses fake, preset, data.
 * The node moves along a predefined mobility trace points specifed in a file.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WC_GPSD_FAKER_MOBI_HPP__
#define WC_GPSD_FAKER_MOBI_HPP__

#include <cstdint>  // uint*_t
#include <string>   // std::string
#include <fstream>  // std::ifstream
#include <vector>   // std::vector

#include "../../util/collections/CircularIterator.hpp" // CircularIterator
#include "../../util/thread/Runnable.hpp"              // class Runnable

#include "WcGpsdFakerFix.hpp" // class GpsDaemonFix


class WcGpsdFakerMobi : public WcGpsdFakerFix {

public:
  /**
   * Constructor. Initializes all the necessary state.
   *
   * @param mobiFname: mobility trace filename.
   */
  WcGpsdFakerMobi(std::string const& mobiFname);
  
  /**
   * Empty destructor.
   */
  virtual ~WcGpsdFakerMobi() override;
  
  /**
   * Move along the mobility trace exactly one step.
   */
  void stepMobiTrace();

private:
  
  /**
   * A vector of GpsInfos.
   */
  typedef std::vector<GpsInfo> GpsInfoVec;
  
  GpsInfoVec mobiTrace;

  CircularIterator<GpsInfoVec> mobiTraceItr;
  
  /**
   * Helper that loads mobility trace from file.
   *
   * @param mobiFname: mobility trace filename.
   * @return the loaded mobility trace.
   */
  GpsInfoVec loadMobiFile(std::string const& mobiFname);

  
public:
  /**
   * Helper class that notifies the gps daemon everytime a mobility trace step needs to be taken.
   */
  class MobiStepper : public Runnable {

    public:
    /**
     * Constructor method.
     *
     * @param gpsdFakerMobi: the gps daemon to the notified everytime a mobility trace step needs to be
     *                       taken.
     */
    MobiStepper(WcGpsdFakerMobi& gpsdFakerMobi);
    
    /**
     * Executes a loop that takes a step on the mobility trace everytime the user hits enter.
     *
     * Overrides Runnable::run().
     * Meant to be run as a thread.
     */
    void run() override;

  private:
    WcGpsdFakerMobi& gpsdFakerMobi;
  };
};

#endif // WC_GPSD_FAKER_MOBI_HPP__
