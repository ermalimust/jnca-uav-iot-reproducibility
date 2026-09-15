/**
 * Implements factory that creates WiFi standard enum values out of strings.
 *
 * Rui Meireles (@vassar.edu) 2024
 */
#include <string>        // std::string
#include <sstream>       // std::stringstream
#include <memory>        // std::unique_ptr, std::make_unique
#include <unordered_map> // std::unordered_map

#include "../../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../../util/log/LogFile.hpp"       // LOG_*
#include "NperfSumStratAm.hpp"                 // class NperfSumStratAm
#include "NperfSumStratEma.hpp"                // class NperfSumStratEma

#include "NperfSumStratFactory.hpp" // class NperfSumStratFactory

#define TPUT_SUMMARIZATION_STRAT_DEF "am" // arithmetic mean
#define EMA_NEW_SAMPLE_WEIGHT_DEF 0.2     // default alpha for exp moving avg

/**
 * Creates and returns a NperfSumStrat object out of the info in the system configuration file.
 *
 * @return a NperfSumStrat corresponding to the system's configuration.
 */
std::unique_ptr<NperfSumStrat> NperfSumStratFactory::create() {

  std::unique_ptr<NperfSumStrat> stratPtr; // return value
  
  // read summarization strategy
  WcConfigFile configFile;
  std::string section("wichoicemaker-tput-summarization");
  std::string stratStr = configFile.strValue(section,
                                             "strategy",
                                             TPUT_SUMMARIZATION_STRAT_DEF);

  // use a switch to distinguish between the different options
  const static std::unordered_map<std::string,int> stratToCase {{"am", 1},
                                                                {"ema", 2}};

  switch (stratToCase.count(stratStr) ? stratToCase.at(stratStr) : 0) {

    case 0: // not found
      { // needed to constraint stringstream scope
        std::stringstream ss;
        ss << "Config exception: section=" << section  << ", " <<
              "value=strategy. " <<
              "Invalid value " << stratStr <<
              " (only 'am' and 'ema' accepted). " <<
              "Defaulting to 'am'.";
        LOG_ERR(ss.str().c_str());
      }
      // no break so we fall through to case 1

    case 1: // am
      stratPtr = std::make_unique<NperfSumStratAm>();
      break;
    case 2: // ema
      // read new sample weight for ema
      const double newSampWeight =
                               configFile.doubleValue(section,
                                                      "ema-new-sample-weight",
                                                      EMA_NEW_SAMPLE_WEIGHT_DEF,
                                                      0,
                                                      1);

      stratPtr = std::make_unique<NperfSumStratEma>(newSampWeight);
      break;
  }
  
  return stratPtr;
}
