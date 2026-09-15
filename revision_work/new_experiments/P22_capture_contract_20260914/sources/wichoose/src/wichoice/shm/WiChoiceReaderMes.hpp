/**
 * Defines class that provides a convenient way to read interface choice from the shared memory region.
 * This class is specific for the interface choice made based on throughput measurements.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_READER_MES_HPP__
#define WI_CHOICE_READER_MES_HPP__

#include "WiChoiceReader.hpp" // class WiChoiceReader

class WiChoiceReaderMes : public WiChoiceReader {

public:
  /**
   * Empty constructor.
   */
  WiChoiceReaderMes();

protected:
  /**
   * Initializes wiChoice field according to the choice we want to work with (i.e., from measurements).
   * Overrides abstract method from WiChoiceReader.
   */
  void initWiChoice() override;

};

#endif // WI_CHOICE_READER_MES_HPP__
