/**
 * Defines class for a GPS daemon that uses fake, preset, data.
 * The node never moves. It stays fixed at the location indicated in the configuration file.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WC_GPSD_FAKER_FIX_HPP__
#define WC_GPSD_FAKER_FIX_HPP__

#include <cstdint> // uint*_t

#include "../GpsDaemon.hpp" // class GpsDaemon

class WcGpsdFakerFix : public GpsDaemon {
  
public:
  
  /**
   * Empty constructor. Initializes all the necessary state.
   */
  WcGpsdFakerFix();
  
  /**
   * Empty destructor.
   */
  virtual ~WcGpsdFakerFix() override;
  
protected:
  
  /**
   * Updates curGpsInfo field according to the latest systime. Everything else is constant.
   * Overrides same-name method from superclass.
   */
  virtual void updateGpsInfo() override;
  
  /**
   * Helper method that pauses execution until the system time changes to the next second.
   *
   * @return the timestamp after waking up from sleep, in seconds.
   */
  uint32_t sleepUntilNextSec();
};

#endif // WC_GPSD_FAKER_FIX_HPP__
