/**
 * Abstract class for objects that can be run and stopped.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "../log/LogFile.hpp" // LOG_*

#include "Runnable.hpp" // class Runnable

/**
 * Empty constructor. Initializes endProgram to false.
 */
Runnable::Runnable() : endProgram(false) { }

/**
 * Signals the work thread to stop executing.
 * Can be overidden, but a default implementation that simply sets endProgram to true  is provided.
 */
void Runnable::stop() {  // what a killer method!
  LOG_MSG("program killed");
  this->endProgram.store(true);
}
