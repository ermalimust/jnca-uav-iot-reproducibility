/**
 * Implements class that provides a convenient way to read interface choice from the shared memory region.
 * This class is specific for the interface choice made based on throughput measurements.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "WiChoiceReaderMes.hpp" // class WiChoiceReaderMes

/**
 * Empty constructor.
 */
WiChoiceReaderMes::WiChoiceReaderMes() : WiChoiceReader() { this->init(); }

/**
 * Initializes wiChoice field according to the choice we want to work with (i.e., from measurements).
 * Overrides abstract method from WiChoiceReader.
 */
void WiChoiceReaderMes::initWiChoice() {
  this->wiChoice = &this->wiChoiceShm->wiChoiceMes;
}
