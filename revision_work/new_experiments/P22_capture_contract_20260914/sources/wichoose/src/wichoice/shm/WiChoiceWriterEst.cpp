/**
 * Implements class that provides a convenient  way to write (and read) the selected interface from the shared
 * memory region. This class is specific for the interface choice made based on throughput estimates.

 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "WiChoiceWriterEst.hpp" // class WiChoiceWriterEst

/**
 * Empty constructor.
 */
WiChoiceWriterEst::WiChoiceWriterEst() : WiChoiceWriter() { this->init(); }

/**
 * Initializes wiChoice field according to the choice we want to work with (i.e., from estimates).
 * Overrides abstract method from WiChoiceReader.
 */
void WiChoiceWriterEst::initWiChoice() {
  this->wiChoice = &this->wiChoiceShm->wiChoiceEst;
}
