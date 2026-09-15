/**
 * Class that implements a GPS daemon using real data from a serial GPS device, by reading the NMEA
 * sentences it produces.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#ifndef WC_GPSD_HPP__
#define WC_GPSD_HPP__

#include "../GpsDaemon.hpp" // class GpsDaemon

#define NMEA_MAX_BUFLEN  100
#define NMEA_MAX_WORDS   30
#define NMEA_WORD_SIZE   30

class WcGpsd : public GpsDaemon {

public:
  /**
   * Empty constructor. Initializes all the necessary state.
   */
  WcGpsd();

protected:
  /**
   * Updates curGpsInfo field accord to the latest systime. Everything else is constant.
   */
  void updateGpsInfo() override;

private:
  enum NmeaType { RMC, GGA, GSA, GSV, OTHER }; // relevant NMEA sentences

  int serialFd; // serial port file descriptor
  
  float headUpdMinSpeed; // min speed required to trigger a heading update
  float posUpdMinSpeed;  // min speed required to trigger a position update

  /**
   * Reads  a NMEA sentence from the serial file descriptor into a string array that is easy to process.
   *
   * @param nmeaType: pointer to variable where type of sentence read is to be stored (output argument).
   * @param darray: pointer to array where NMEA data is to be stored (output argument).
   * @return true on success, false on checksum fail.
   */
  bool readNmea(NmeaType *nmeaType, char darray[][NMEA_WORD_SIZE]);
};

#endif // WC_GPSD_HPP__
