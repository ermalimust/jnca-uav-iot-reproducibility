/**
 * Defines class that provides a convenient  way to write (and read) the selected interface from the shared
 * memory region. This class is specific for the interface choice made based on throughput estimates.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_WRITER_EST_HPP__
#define WI_CHOICE_WRITER_EST_HPP__

#include "WiChoiceWriter.hpp" // class WiChoiceWriter

class WiChoiceWriterEst : public WiChoiceWriter {

public:
  /**
   * Empty constructor.
   */
  WiChoiceWriterEst();

protected:
  /**
   * Initializes wiChoice field according to the choice we want to work with (i.e., from estimates).
   * Overrides abstract method from WiChoiceReader.
   */
  void initWiChoice() override;
};

#endif // WI_CHOICE_WRITER_EST_HPP__
