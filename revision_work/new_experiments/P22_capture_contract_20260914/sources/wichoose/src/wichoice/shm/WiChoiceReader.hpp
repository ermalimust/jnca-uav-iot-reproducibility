/**
 * Defines class that provides a convenient way to read interface choice from the shared memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_READER_HPP__
#define WI_CHOICE_READER_HPP__

#include <string> // std::string

#include "../../util/shm/ShmReader.hpp" // class ShmReader

#include "../WiChoice.hpp" // struct WiChoice
#include "WiChoiceShm.hpp" // struct WiChoiceShm

class WiChoiceReader : virtual public ShmReader {

public:
  /**
   * Empty constructor.
   */
  WiChoiceReader();

  /**
   * Returns currently selected interface.
   *
   * @return currently selected interface.
   */
  const std::string getIfaceNow();

  /**
   * Blocks waiting for selected interface to be updated. As soon as it does, it returns it.
   *
   * @return interface that has just been updated.
   */
  const std::string getIfaceOnUpdate();

  /**
   * Returns copy of current wifi choice.
   *
   * @return copy of current wifi choice.
   */
  const WiChoice getWiChoiceNow();
  
  /**
   * Blocks waiting for wifi choice to be updated. As soon as it does, it returns a copy of it.
   *
   * @return copy of just-updated wifi choice.
   */
  const WiChoice getWiChoiceOnUpdate();

protected:
  WiChoiceShm* wiChoiceShm; // pointer to shared memory region holding gps info
  WiChoice* wiChoice;

  /**
   * Helper method that reads and loads configuration parameters (shm path and )
   * from the config file.
   */
  virtual void readConfig() override;

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
  WiChoiceReader(const bool init);

  /**
   * Returns the iface name in the shared memory either now or later
   * when there's an update, depending on the boolean argument.
   *
   * @param onUpdate: whether we should wait for an update before reading or not.
   */
  const std::string getIface(bool onUpdate);
  
  /**
   * Returns a copy of the complete wifi choice (iface and tstamp), either now or right after the choice is next
   * updated, depending on the boolean argument.
   *
   * @param onUpdate: whether we should wait for an update before reading or not.
   * @return the copied wifi choice.
   */
  const WiChoice getWiChoice(bool onUpdate);

  /**
   * Initializes wiChoice field according to the choice we want to work with (i.e., from measurements or
   * estimates). Concrete subclasses must implement.
   */
  virtual void initWiChoice() = 0;

};

#endif // WI_CHOICE_READER_HPP__
