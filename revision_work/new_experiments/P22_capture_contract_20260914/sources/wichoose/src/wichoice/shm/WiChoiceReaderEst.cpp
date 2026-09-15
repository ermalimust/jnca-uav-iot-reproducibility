/**
 * Implements class that provides a convenient way to read interface choice from the shared memory region.
 * This class is specific for the interface choice made based on throughput estimates.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "WiChoiceReaderEst.hpp" // class WiChoiceReaderEst

/**
 * Empty constructor.
 */
WiChoiceReaderEst::WiChoiceReaderEst() : WiChoiceReader() { this->init(); }

/**
 * Initializes wiChoice field according to the choice we want to work with (i.e., from estimates).
 * Overrides abstract method from WiChoiceReader.
 */
void WiChoiceReaderEst::initWiChoice() {
  this->wiChoice = &this->wiChoiceShm->wiChoiceEst;
}
