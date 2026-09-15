/**
 * Defines class that lets us manually set the wifi interface choice.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef WI_CHOICE_FAKER_HPP__
#define WI_CHOICE_FAKER_HPP__

#include <set>    // std::set
#include <string> // std::string

#include "../../util/thread/Runnable.hpp" // class Runnable
#include "../shm/WiChoiceWriter.hpp"      // class WiChoiceWriter

class WiChoiceFaker : public Runnable {

public:
  
  /**
   * Empty constructor.
   *
   * @param wiChoiceWriter: object to use to write wifi choice to shared memory.
   */
  WiChoiceFaker(WiChoiceWriter& wiChoiceWriter);

  /**
   * Executes a loop of reading an interface name from the standard input, and setting it as the
   * the choice for the overall system.
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;

private:
  WiChoiceWriter& wiChoiceWriter;
  std::set<std::string> validIfacesSet;

};

#endif // WI_CHOICE_FAKER_HPP__
