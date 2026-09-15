/**
 * General-purpose class to read configuration parameters from a text file.
 *
 * René Nyffenegger {rene.nyffenegger@adp-gmbh.ch}
 * Rui Meireles (@vassar.edu), 2023, 2024
 */

#ifndef CONFIG_FILE_HPP__
#define CONFIG_FILE_HPP__

#include <cstdint> // uint*_t
#include <string>  // std::string
#include <map>     // std::map

class ConfigFile {
  
public:
  /**
   * Constructor method. Loads up config from file with provided name.
   *
   * @param fname: filename of file to load configuration from.
   */
  ConfigFile(std::string const& fname);
  
  /**
   * Attempts to read and return a string value associated with provided section and entry. If an error occurs,
   * the provided default value is returned.
   *
   * @param section: name of config section to read from.
   * @param entry: name of entry within config section to read from.
   * @param defValue: the value to use in the event of an error.
   * @return string value associated with (section, entry) combination, or defValue if an error occurs.
   */
  std::string const& strValue(std::string const& section,
                              std::string const& entry,
                              std::string const& defValue);
  
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
  uint8_t ubyteValue(std::string const& section,
                     std::string const& entry,
                     const uint8_t defValue) const;
  
  /**
   * Attempts to read and return an integer value associated with provided section and entry. If an error
   * occurs, the provided default value is returned.
   *
   * @param section: name of config section to read from.
   * @param entry: name of entry within config section to read from.
   * @param defValue: the value to use in the event of an error.
   * @return integer value associated with (section, entry) combination, or defValue if an error occurs.
   */
  int intValue(std::string const& section,
               std::string const& entry,
               const int defValue) const;
  
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
  unsigned int uintValue(std::string const& section,
                         std::string const& entry,
                         const unsigned int defValue) const;
  
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
  unsigned int uintValue(std::string const& section,
                         std::string const& entry,
                         const unsigned int defValue,
                         const unsigned int minValue,
                         const unsigned int maxValue) const;
  
  /**
   * Attempts to read and return a long long integer value associated with provided section and entry.
   * If an error occurs, the provided default value is returned.
   *
   * @param section: name of config section to read from.
   * @param entry: name of entry within config section to read from.
   * @param defValue: the value to use in the event of an error.
   * @return long long value associated with (section, entry) combination, or defValue if an error occurs.
   */
  long long longLongValue(std::string const& section,
                          std::string const& entry,
                          const long long defValue) const;
  
  /**
   * Attempts to read and return a float value associated with provided section and entry.
   * If an error occurs, the provided default value is returned.
   *
   * @param section: name of config section to read from.
   * @param entry: name of entry within config section to read from.
   * @param defValue: the value to use in the event of an error.
   * @return float value associated with (section, entry) combination, or defValue if an error occurs.
   */
  float floatValue(std::string const& section,
                   std::string const& entry,
                   const float defValue) const;
  
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
  float floatValue(std::string const& section,
                   std::string const& entry,
                   const float defValue,
                   const float minValue,
                   const float maxValue) const;
  
  /**
   * Attempts to read and return a double value associated with provided section and entry.
   * If an error occurs, the provided default value is returned.
   *
   * @param section: name of config section to read from.
   * @param entry: name of entry within config section to read from.
   * @param defValue: the value to use in the event of an error.
   * @return double value associated with (section, entry) combination, or defValue if an error occurs.
   */
  double doubleValue(std::string const& section,
                     std::string const& entry,
                     const double defValue) const;
  
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
  double doubleValue(std::string const& section,
                     std::string const& entry,
                     const float defValue,
                     const float minValue,
                     const float maxValue) const;
  
protected:
  typedef std::map<std::string, std::string> EntryMap;
  typedef std::map<std::string, EntryMap> SectionMap;
  SectionMap sectionMap;
  
  /**
   * Attempts to read (string) value associated with provided section and entry.
   *
   * @param section: name of config section to read from.
   * @param entry: name of entry within config section to read from.
   * @throws std::runtime\_error if provided (section, entry) combination does not exist.
   * @return (string) value associated with provided (section, entry) combination.
   */
  std::string const& value(std::string const& section,
                           std::string const& entry) const;
  
  /**
   * Helper that logs a fatal error message to log file associated with the running process.
   *
   * @param section: config section the error pertains to.
   * @param entry: section entry the error pertains to.
   * @param error: error message.
   */
  void logErrorFatal(std::string const& section,
                     std::string const& entry,
                     std::string const& error) const;
  
  /**
   * Helper that logs a non-fatal error message to log file associated with the running process.
   * The mitigation strategy is passed in as an argument.
   *
   * @param section: config section the error pertains to.
   * @param entry: section entry the error pertains to.
   * @param error: error message.
   * @param mitigation: how the error will be mitigated.
   */
  void logError(std::string const& section,
                std::string const& entry,
                std::string const& error,
                std::string const& mitigation) const;
  
  /**
   * Helper that logs a non-fatal error message to log file associated with the running process, where the
   * mitigation strategy is assuming a default value.
   *
   * @param section: config section the error pertains to.
   * @param entry: section entry the error pertains to.
   * @param error: error message.
   * @param defValue: value that will be assumed for (section, entry) given the error.
   */
  void logErrorDef(std::string const& section,
                   std::string const& entry,
                   std::string const& error,
                   std::string const& defValue) const;
  
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
  void logErrorDef(std::string const& section,
                   std::string const& entry,
                   std::string const& error,
                   std::string const& defValue,
                   std::string const& minValue,
                   std::string const& maxValue) const;

/**
  * Converts string to unsigned 8-bit value.
  *
  * @param str: the string to convert.
  * @param uint: reference to where the unsigned value should be written to.
  * @return true if the conversion was successful, false otherwise.
  */
  bool strToUint8(std::string const& str, uint8_t& uint) const;

};

#endif // CONFIG_FILE_HPP__
