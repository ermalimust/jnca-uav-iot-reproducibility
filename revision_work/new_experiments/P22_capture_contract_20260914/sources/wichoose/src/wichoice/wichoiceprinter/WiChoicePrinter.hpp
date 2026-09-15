/**
 * Defines class that prints the interface selection to the standard output, every time a new one is made.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef WI_CHOICE_PRINTER_HPP__
#define WI_CHOICE_PRINTER_HPP__

#include <string> // std::string

#include "../../util/thread/Runnable.hpp" // class Runnable
#include "../shm/WiChoiceReader.hpp"      // class WiChoiceReader

class WiChoicePrinter : public Runnable {

public:
  /**
   * Empty constructor.
   *
   * @param wiChoiceReader: where to get wichoice information from.
   * @param printTag: the string to tag each printed wichoice with.
   * @param printHeader: boolean indicating whether we should print a header at the start or not.
   */
  WiChoicePrinter(WiChoiceReader& wiChoiceReader,
                  std::string const& printTag,
                  const bool printHeader);

  WiChoicePrinter(WiChoiceReader& wiChoiceReader);

  /**
   * Executes a loop of reading new interface information until the endProgram flag becomes true
   * (through some external intervention).
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;

  /**
   * Halts the runnable. The superclass one is insufficient because the thread may be blocked waiting for
   * the wichoice to be updated. We must therefore trigger that update manually.
   */
  virtual void stop() override;
  
private:
  WiChoiceReader& wiChoiceReader;
  const std::string printTag;
};

#endif // WI_CHOICE_PRINTER_HPP__
