/**
 * Defines class that provides a convenient way to read GPS information from the shared memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef GPS_INFO_READER_HPP__
#define GPS_INFO_READER_HPP__

#include <cstdint> // uint*_t

#include "../../util/shm/ShmReader.hpp" // class ShmReader
#include "../GpsInfo.hpp"               // struct GpsInfo
#include "GpsInfoShm.hpp"               // struct GpsInfoShm

class GpsInfoReader : virtual public ShmReader {

public:
  /**
   * Empty constructor. It always initializes the reader.
   */
  GpsInfoReader();
    
  /**
   * Copies GPS information from the shared memory to the structure passed in as an argument, right now.
   *
   * @param gpsInfo: reference of the the gps info struct to copy to.
   */
  void getGpsInfoNow(GpsInfo& gpsInfo);

  /**
   * Copies GPS information from the shared memory to the structure passed in as an argument,
   * but only when the data is updated by the GPS daemon. It blocks waiting until then.
   *
   * @param gpsInfo: reference of the the gps info struct to copy to.
   */
  void getGpsInfoOnUpdate(GpsInfo& gpsInfo);
  
  /**
   * Retrieve and return the current GPS time.
   *
   * @return current GPS time.
   */
  uint32_t getGpstimeNow() const;
  
  /**
   * Block waiting for the GPS time to be updated, then return its new value.
   *
   * @return recently updated GPS time.
   */
  uint32_t getGpstimeOnUpdate();


protected:
  GpsInfoShm* gpsInfoShm; // pointer to shared memory region holding gps info
  
  /**
   * Helper method that reads and loads configuration parameters (shm path and timezone)
   * from the config file.
   */
  void readConfig() override;
  
  /**
   * High-level helper that is meant to put the shared memory in a usable state, after the config has been
   * read.
   */
  virtual void initShm() override;
  
  /**
   * Constructor that initializes the reader or not, depending on the boolean
   *
   * @param init: if true, the reader is initialized, otherwise it is not.
   */
  GpsInfoReader(const bool init);

private:
  /**
   * Helper for the public getter methods. It  copies GPS information from the shared memory to the
   * structure passed in as an argument either now or later when there's an update, depending on the
   * boolean argument.
   *
   * @param gpsInfo: reference of the the gps info struct to copy to.
   * @param onUpdate: whether we should wait for an update before copying or not.
   */
  void getGpsInfo(GpsInfo& gpsInfo, bool onUpdate);
};

#endif // GPS_INFO_READER_HPP__
