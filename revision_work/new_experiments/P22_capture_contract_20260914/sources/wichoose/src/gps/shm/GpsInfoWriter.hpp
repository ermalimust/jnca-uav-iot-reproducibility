/**
 * Defines class that provides a convenient way to write (and read) GPS information from the shared
 * memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef GPS_INFO_WRITER_HPP__
#define GPS_INFO_WRITER_HPP__

#include "../../util/shm/ShmWriter.hpp" // class ShmWriter
#include "../GpsInfo.hpp"               // struct GpsInfo
#include "GpsInfoReader.hpp"            // class GpsInfoReader

class GpsInfoWriter : public GpsInfoReader, public ShmWriter {

public:
  /**
   * Empty constructor.
   */
  GpsInfoWriter();

  /**
   * High-level helper that is meant to put the shared memory in a usable state, after the config has been
   * read.
   */
  void initShm() override;
  
  /**
   * Copies GPS information from the struct passed in as argument to shared memory region.
   *
   * @param gpsInfo: reference to the gps info struct to copy from.
   */
  void setGpsInfo(GpsInfo const& gpsInfo);
};

#endif // GPS_INFO_WRITER_HPP__
