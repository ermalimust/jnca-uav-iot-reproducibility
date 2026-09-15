/**
 * Implements class that lets us manually set the wifi interface choice.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <sstream>  // std::stringstream
#include <iostream> // std::in, std::cout, std::cerr, std::endl
#include <string>   // std::string, std::getline
#include <memory>   // std::unique_ptr, std::make_unique
#include <set>      // std::set

#include "../shm/WiChoiceWriter.hpp"        // class WiChoiceWriter
#include "../shm/WiChoiceWriterMes.hpp"     // class WiChoiceWriterMes
#include "../shm/WiChoiceWriterEst.hpp"     // class WiChoiceWriterEst
#include "../../util/log/LogFile.hpp"       // LOG_*
#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/conf/Network.hpp"      // struct Network, type NetMap
#include "../../util/thread/Runner.hpp"     // class Runner

#include "WiChoiceFaker.hpp" // class WiChoiceFaker

#define LOG_FNAME "/var/log/wichoicefaker.log"

/**
 * Empty constructor.
 */
WiChoiceFaker::WiChoiceFaker(WiChoiceWriter& wiChoiceWriter) :
                             wiChoiceWriter(wiChoiceWriter), validIfacesSet() {

  // read configuration for valid interface names
  WcConfigFile configFile;
  NetworkMap netMap;
  configFile.networks("wichoice", netMap); // read networks
  for (auto& kvp : netMap) { // get iface for each network
    Network const& net = kvp.second;
    this->validIfacesSet.emplace(net.iface);
  }
}

/**
 * Executes a loop of reading an interface name from the standard input, and setting it as the
 * the choice for the overall system.
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void WiChoiceFaker::run() {
  
  // create a string with all the valid choices
  std::stringstream ss;
  ss << "Valid choices: ";
  for (auto& iface : this->validIfacesSet) ss << iface << " ";
  const std::string validIfacesStr = ss.str();
  
  std::cout << "Welcome to WiChoiceFaker. " 
             << "Current iface: " << this->wiChoiceWriter.getIfaceNow()
             << ". " << validIfacesStr << std::endl;
  std::cout << "To exit press Ctrl-C, then Enter." << std::endl;

  std::string inIface;
  while (!this->endProgram.load()) { // main loop

    std::cout << std::endl << "Choice: ";
    
    if (!std::getline(std::cin, inIface) || inIface.size() == 0)
      continue; // error upon exit signal
    
    if (this->validIfacesSet.count(inIface) > 0) { // valid input
      this->wiChoiceWriter.setIface(inIface);
      
      std::cout << "Chosen: " << inIface << std::endl;
      
    } else { // invalid input
      std::cout << "Invalid input. " << validIfacesStr << std::endl;
    }
    
  } // main loop end
}

// WiChoiceFaker class implementation end

/**
 * The procedure that actually bootstraps the program.
 *
 * @param argc: number of command line arguments, including the program's name.
 * @param argv: array of command line arguments, as strings.
 * @return 0 on success, other value on error.
 */
int main(int argc, char *argv[]) {

  // what type of choice are we making?
  if (argc < 2) { // need a second argument
    std::cerr << "Need argument: mes (measured) or est (estimated)"
              << std::endl;
    return 1;
  }
  std::string typeArg(argv[1]); // save choice as std::string

  LOG_INIT(LOG_FNAME, "wichoicefaker");

  LOG_MSG("wichoicefaker starting");

  // initialize wiChoiceWriterPtr
  auto wiChoiceWriterPtr = std::unique_ptr<WiChoiceWriter>{};
  
  const std::set<std::string> mesArgs = {"mes", "measured"};
  const std::set<std::string> estArgs = {"est", "estimated"};
  if (mesArgs.count(typeArg))
    wiChoiceWriterPtr = std::make_unique<WiChoiceWriterMes>();
  else if (estArgs.count(typeArg))
    wiChoiceWriterPtr = std::make_unique<WiChoiceWriterEst>();
  else {
    std::cerr << "Invalid argument " << typeArg
              << ". Only mes (measured) and est (estimated) supported."
              << std::endl;
    LOG_CLOSE();
    return 2;
  }
  
  if (argc > 2) // we ignore spurious arguments
    std::cerr << "Ignoring all command-line arguments beyond the first."
              << std::endl;

  WiChoiceFaker wiChoiceFaker(*wiChoiceWriterPtr); // create runnable

  Runner& runner = Runner::getInstance(); // get runner
  runner.addRunnable(wiChoiceFaker);      // add runnable to runner
  runner.run();                           // actually run the runnable

  LOG_CLOSE();

  return 0;
}
