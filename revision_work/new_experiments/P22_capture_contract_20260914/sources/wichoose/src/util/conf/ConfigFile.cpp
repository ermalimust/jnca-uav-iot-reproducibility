/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 *  General-purpose class to read configuration parameters from a text file.
 *
 * René Nyffenegger {rene.nyffenegger@adp-gmbh.ch}
 * Rui Meireles (@vassar.edu), 2023, 2024
 *  */

#include <fstream>   // std::ifstream
#include <sstream>   // std::stringstream
#include <string>    // std::string, std::to_string, std::sto*, std::getline
#include <stdexcept> // std::exception, std::runtime_error

#include "../log/LogFile.hpp" // LOG_*

#include "ConfigFile.hpp" // class ConfigFile

/**
 * Helper method that trims the provided string according the specified delimiters.
 *
 * @param source: the string to be trimmed.
 * @param delims: the characters that delimit the trimming.
 * @return the trimmed string.
 */
std::string trim(std::string const& source, char const* delims = " \t\r\n") {
	
  std::string result(source);
  std::string::size_type index = result.find_last_not_of(delims);
	
  if (index != std::string::npos) result.erase(++index);
	
  index = result.find_first_not_of(delims);
  
  if (index != std::string::npos) result.erase(0, index);
  else result.erase();

  return result;
}

/**
 * Constructor method. Loads up config from file with provided name.
 *
 * @param fname: filename of file to load configuration from.
 */
ConfigFile::ConfigFile(std::string const& fname) : sectionMap() {

  std::ifstream file(fname.c_str());
  if (not file.good()) {
    std::stringstream ss;
    ss << "Couldn't open config file \"" << fname << "\". Will use defaults.";
    LOG_ERR(ss.str().c_str());
  }

  
  std::string inSection; // defaults to ""
  this->sectionMap[inSection]; // ensures default section is in map
  
  std::string line;
  std::string name;
  std::string value;
  std::size_t posEqual;
  std::size_t posHash;
  while (std::getline(file,line)) {
		
    if (!line.length()) continue;
		
    if (line[0] == '#') continue;
		
    if (line[0] == '[') {
      inSection = trim(line.substr(1,line.find(']')-1));
      this->sectionMap[inSection]; // ensures default section is in map
      continue;
    }
    
    // start by discarding everything from first # onwards (if # exists)
    posHash = line.find('#');
    if (posHash != std::string::npos)
      line = line.substr(0, posHash); // substr of len posHash, starting at 0
    
    posEqual = line.find('=');
    name  = trim(line.substr(0,posEqual));
  
    value = trim(line.substr(posEqual+1));

    this->sectionMap[inSection][name] = value; // save (name, value) in section
  }
  
  file.close();
}


/**
 * Attempts to read (string) value associated with provided section and entry.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @throws std::runtime\_error if provided (section, entry) combination does not exist.
 * @return (string) value associated with provided (section, entry) combination.
 */
std::string const& ConfigFile::value(std::string const& section,
                                     std::string const& entry) const {
	
  auto sectionItr = this->sectionMap.find(section);
  if (sectionItr == this->sectionMap.end())
    throw std::runtime_error("section does not exist");
  
  EntryMap const& entryMap = sectionItr->second;
  
  auto entryItr = entryMap.find(entry);
  if (entryItr == entryMap.end())
    throw std::runtime_error("entry does not exist");
  
  return entryItr->second;
}

/**
 * Attempts to read and return a string value associated with provided section and entry. If an error occurs,
 * the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @return string value associated with (section, entry) combination, or defValue if an error occurs.
 */
std::string const& ConfigFile::strValue(std::string const& section,
                                     std::string const& entry, 
                                     std::string const& defValue) {
  try {
    return this->value(section, entry);
  } catch (std::exception const& e) {
    this->logErrorDef(section, entry, e.what(), defValue);
  }

  return defValue;
}

/**
 * Attempts to read and return an unsigned 8-bit value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @return unsigned 8-bit value associated with (section, entry) combination, or defValue if an error
 *         occurs.
 */
uint8_t ConfigFile::ubyteValue(std::string const& section,
                               std::string const& entry,
                               const uint8_t defValue) const {

  uint8_t retval = defValue;

  try {
    const std::string strValue = value(section, entry);

    if (!this->strToUint8(strValue, retval))
      this->logErrorDef(section, entry, "value out of range",
                        std::to_string((unsigned int)defValue), "0", "255");

  } catch (std::runtime_error const& e) {
    this->logErrorDef(section, entry, e.what(),
                      std::to_string((unsigned int)defValue));
  }
  return retval;
}

/**
 * Attempts to read and return an integer value associated with provided section and entry. If an error
 * occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @return integer value associated with (section, entry) combination, or defValue if an error occurs.
 */
int ConfigFile::intValue(std::string const& section,
                         std::string const& entry,
                         const int defValue) const {
  
  int retval = defValue;
  
  try {
    const std::string strValue = this->value(section, entry);
    retval = std::stoi(strValue);
  } catch (std::exception const& e) {
    this->logErrorDef(section, entry, e.what(), std::to_string(defValue));
  }
  
  return retval;
}

/**
 * Attempts to read and return an unsigned integer value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @return unsigned integer value associated with (section, entry) combination, or defValue if an error
 *         occurs.
 */
unsigned int ConfigFile::uintValue(std::string const& section,
                                   std::string const& entry,
                                   const unsigned int defValue) const {
  
  unsigned int retval = defValue;
  
  try {
    const std::string strValue = this->value(section, entry);
    retval = (unsigned int) std::stoul(strValue);
  } catch (std::exception const& e) {
    this->logErrorDef(section, entry, e.what(), std::to_string(defValue));
  }
  
  return retval;
}

/**
 * Attempts to read and return a ranged unsigned integer value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @param minValue: smallest acceptable value.
 * @param minValue: largest acceptable value.
 * @return unsigned integer value associated with (section, entry) combination, or defValue if an error
 *         occurs.
 */
unsigned int ConfigFile::uintValue(std::string const& section,
                                   std::string const& entry,
                                   const unsigned int defValue,
                                   const unsigned int minValue,
                                   const unsigned int maxValue) const {
  
  unsigned int retval = this->uintValue(section, entry, defValue);
  
  if (retval < minValue || retval > maxValue) {
    this->logErrorDef(section, entry, "value out of range",
                   std::to_string(defValue), std::to_string(minValue),
                   std::to_string(maxValue));
  }

  return retval;
}

/**
 * Attempts to read and return a long long integer value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @return long long value associated with (section, entry) combination, or defValue if an error
 *         occurs.
 */
long long ConfigFile::longLongValue(std::string const& section,
                                    std::string const& entry,
                                    const long long defValue) const {

  long long retval = defValue;

  try {
    const std::string strValue = this->value(section, entry);
    retval = std::stoll(strValue);
  } catch (std::exception const& e) {
    this->logErrorDef(section, entry, e.what(), std::to_string(defValue));
  }

  return retval;
}



/**
 * Attempts to read and return a float value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @return float value associated with (section, entry) combination, or defValue if an error occurs.
 */
float ConfigFile::floatValue(std::string const& section,
                             std::string const& entry,
                             const float defValue) const {

  float retval = defValue;

  try {
    const std::string strValue = this->value(section, entry);
    retval = std::stof(strValue);
  } catch (std::exception const& e) {
    this->logErrorDef(section, entry, e.what(), std::to_string(defValue));
  }

  return retval;
}

/**
 * Attempts to read and return a ranged float value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @param minValue: smallest acceptable value.
 * @param minValue: largest acceptable value.
 * @return float associated with (section, entry) combination, or defValue if an error occurs.
 */
float ConfigFile::floatValue(std::string const& section,
                             std::string const& entry,
                             const float defValue,
                             const float minValue,
                             const float maxValue) const {

  float retval = this->floatValue(section, entry, defValue);
  
  if (retval < minValue || retval > maxValue) {
    this->logErrorDef(section, entry, "value out of range",
                   std::to_string(defValue), std::to_string(minValue),
                   std::to_string(maxValue));
  }

  return retval;
}

/**
 * Attempts to read and return a double value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @return double value associated with (section, entry) combination, or defValue if an error occurs.
 */
double ConfigFile::doubleValue(std::string const& section,
                               std::string const& entry,
                               const double defValue) const {

  double retval = defValue;

  try {
    const std::string strValue = this->value(section, entry);
    retval = std::stod(strValue);
  } catch (std::exception const& e) {
    this->logErrorDef(section, entry, e.what(), std::to_string(defValue));
  }

  return retval;
}

/**
 * Attempts to read and return a ranged double value associated with provided section and entry.
 * If an error occurs, the provided default value is returned.
 *
 * @param section: name of config section to read from.
 * @param entry: name of entry within config section to read from.
 * @param defValue: the value to use in the event of an error.
 * @param minValue: smallest acceptable value.
 * @param minValue: largest acceptable value.
 * @return double associated with (section, entry) combination, or defValue if an error occurs.
 */
double ConfigFile::doubleValue(std::string const& section,
                               std::string const& entry,
                               const float defValue,
                               const float minValue,
                               const float maxValue) const {
 
  double retval = this->doubleValue(section, entry, defValue);
  
  if (retval < minValue || retval > maxValue) {
    this->logErrorDef(section, entry, "value out of range",
                   std::to_string(defValue), std::to_string(minValue),
                   std::to_string(maxValue));
  }

  return retval;
}

/**
 * Generalized helper that logs an error message to log file associated with the running process.
 *
 * @param section: config section the error pertains to.
 * @param entry: section entry the error pertains to.
 * @param error: error message.
 * @param fatal: whether the error is fatal or not (default false).
 * @param mitigation: how the error will be mitigated (default empty).
 * @param defValue: value that will be assumed for (section, entry) given the error (default empty).
 * @param minValue: the smallest value that (section, entry) can assume (default empty).
 * @param maxValue: the largest value that (section, entry) can assume (default empty).
 */
void logErrorGen(std::string const& section,
                             std::string const& entry,
                             std::string const& error,
                             const bool fatal = false,
                             std::string const& mitigation = "",
                             std::string const& defValue = "",
                             std::string const& minValue = "",
                 std::string const& maxValue = "") {
  
  std::stringstream ss;
  
  // baseline common to all
  ss << "Config exception: section=" << section << ", value=" << entry << ". "
  << "Error: " << error << ".";
  
  if (mitigation != "")      // general mitigation
    ss << " Mitigation: " << mitigation << ".";
  else if (defValue != "") { // def value-based mitigation
    ss << " Mitigation: using default value " << defValue;
    
    if (minValue != "" && maxValue != "")
      ss << " (min= " << minValue << ", max= " << maxValue << ")";
    ss << ".";
  }
  
  if (fatal) { LOG_FATAL_EXIT(ss.str().c_str()); } // macros need the brackets
  else { LOG_ERR(ss.str().c_str()); }
}


/**
 * Helper that logs a fatal error message to log file associated with the running process.
 *
 * @param section: config section the error pertains to.
 * @param entry: section entry the error pertains to.
 * @param error: error message.
 */
void ConfigFile::logErrorFatal(std::string const& section,
                               std::string const& entry,
                               std::string const& error) const {
  
  logErrorGen(section, entry, error, true /*fatal*/);
}

/**
 * Helper that logs a non-fatal error message to log file associated with the running process.
 * The mitigation strategy is passed in as an argument.
 *
 * @param section: config section the error pertains to.
 * @param entry: section entry the error pertains to.
 * @param error: error message.
 * @param mitigation: how the error will be mitigated.
 */
void ConfigFile::logError(std::string const& section, 
                          std::string const& entry,
                          std::string const& error,
                          std::string const& mitigation) const {
  
  logErrorGen(section, entry, error,
              false /*fatal*/,
              mitigation);
}

/**
 * Helper that logs a non-fatal error message to log file associated with the running process, where the
 * mitigation strategy is assuming a default value.
 *
 * @param section: config section the error pertains to.
 * @param entry: section entry the error pertains to.
 * @param error: error message.
 * @param defValue: value that will be assumed for (section, entry) given the error.
 */
void ConfigFile::logErrorDef(std::string const& section,
                             std::string const& entry,
                             std::string const& error,
                             std::string const& defValue) const {
  
  logErrorGen(section, entry, error,
              false /*fatal*/,
              "" /*mitigation*/,
              defValue);
}

/**
 * Helper that logs a non-fatal error message to log file associated with the running process, where the
 * mitigation strategy is assuming a default value. This specific version targets numeric entries that have a
 * fixed range of possible values, which is to be communicated to the user..
 *
 * @param section: config section the error pertains to.
 * @param entry: section entry the error pertains to.
 * @param error: error message.
 * @param defValue: value that will be assumed for (section, entry) given the error.
 * @param minValue: the smallest value that (section, entry) can assume.
 * @param maxValue: the largest value that (section, entry) can assume.
 */
void ConfigFile::logErrorDef(std::string const& section,
                             std::string const& entry,
                             std::string const& error,
                             std::string const& defValue,
                             std::string const& minValue,
                             std::string const& maxValue) const {

  logErrorGen(section, entry, error,
              false /*fatal*/,
              "" /*mitigation*/,
              defValue,
              minValue,
              maxValue);
}


/**
 * Converts string to unsigned 8-bit value.
 *
 * @param str: the string to convert.
 * @param uint: reference to where the unsigned value should be written to.
 * @return true if the conversion was successful, false otherwise.
 */
bool ConfigFile::strToUint8(std::string const& str, uint8_t& uint) const {
  
  bool retval = true;
  try {
    unsigned u = std::stoul(str);
    if (u <= 255) uint = u; else retval = false;
  } catch (std::exception const& e) { retval = false; }

  return retval;
}
