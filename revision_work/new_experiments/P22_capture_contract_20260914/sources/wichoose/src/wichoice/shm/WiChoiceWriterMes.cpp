/**
 * Implements class that provides a convenient  way to write (and read) the selected interface from the shared
 * memory region. This class is specific for the interface choice made based on throughput measurements.

 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "WiChoiceWriterMes.hpp" // class WiChoiceWriterMes

/**
 * Empty constructor.
 */
WiChoiceWriterMes::WiChoiceWriterMes() : WiChoiceWriter() { this->init(); }

/**
 * Initializes wiChoice field according to the choice we want to work with (i.e., from measurements).
 * Overrides abstract method from WiChoiceReader.
 */
void WiChoiceWriterMes::initWiChoice() {
  this->wiChoice = &this->wiChoiceShm->wiChoiceMes;
}
