/**
 * Defines class that provides a convenient way to read interface choice from the shared memory region.
 * This class is specific for the interface choice made based on throughput estimates.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_READER_EST_HPP__
#define WI_CHOICE_READER_EST_HPP__

#include "WiChoiceReader.hpp" // class WiChoiceReader

class WiChoiceReaderEst : public WiChoiceReader {

public:
  /**
   * Empty constructor.
   */
  WiChoiceReaderEst();

protected:
  /**
   * Initializes wiChoice field according to the choice we want to work with (i.e., from estimates).
   * Overrides abstract method from WiChoiceReader.
   */
  void initWiChoice() override;

};

#endif // WI_CHOICE_READER_EST_HPP__
