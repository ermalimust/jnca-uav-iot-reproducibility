/**
 * Defines class that provides a convenient  way to write (and read) the selected interface from the shared
 * memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_WRITER_HPP__
#define WI_CHOICE_WRITER_HPP__

#include <string> // std::string

#include "../../util/shm/ShmWriter.hpp"    // class ShmWriter
#include "../../gps/shm/GpsInfoReader.hpp" // class GpsInfoReader

#include "WiChoiceReader.hpp"        // class WiChoiceReader

class WiChoiceWriter : public WiChoiceReader, public ShmWriter {

public:
  /**
   * Empty constructor.
   */
  WiChoiceWriter();

  /**
   * Sets shared memory iface to be equal to that passed in as an argument. If the interface is different
   * from the one already in the shared memory, listeners are notified that an update has taken place.
   *
   * @param iface: reference of the string to copy from. Its length must not be larger than
   *  MAX_IFACE_NAME_LENGTH, otherwise an error will be thrown.
   *
   * @return true if the iface actually changed from the previous value, and false otherwise.
   */
  bool setIface(std::string const& iface);

protected:
  /**
   * Helper method that reads and loads configuration parameters (shm path and default interface)
   * from the config file.
   */
  void readConfig() override;

  /**
   * High-level helper that is meant to put the shared memory in a usable state, after the config has been
   * read.
   */
  void initShm() override;

private:
  std::string ifaceDef;        // default interface to use at the start
  GpsInfoReader gpsInfoReader; // to timestamp the choices
};

#endif // WI_CHOICE_WRITER_HPP__
