/**
 * Defines a data structure designed to represent an interface choice.
 * The structure must be usable within a shared memory region, so it must be a POD (Plain Old Data)
 * type (only have PODs as members, no user-defined destructor, no user-defined copy assignment operator,
 * and no nonstatic members of pointer-to-member type).
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_HPP__
#define WI_CHOICE_HPP__

#include <cstdint> // uint*_t
#include <string>  // std::string

#define MAX_IFACE_NAME_LENGTH 16 // linux doesn't let ifaces have more

struct WiChoice {
  char iface[MAX_IFACE_NAME_LENGTH+1]; // chosen interface
  uint32_t tstamp;                     // time at which choice was made (secs)
  
  /**
   * Return struct's iface, as an std::string.
   *
   * @return the struct's iface, as an std::string.
   */
  std::string getIface();
  
  /**
   * Sets interface name to be equal to that passed in as an argument.
   *
   * @param iface: reference of the string to copy from. Its length must not be larger than
   *  MAX_IFACE_NAME_LENGTH, otherwise an error will be thrown.
   * @return true if iface changed as a result of this call, and false otherwise.
   */
  bool setIface(std::string const& iface);
};
 
#endif // WI_CHOICE_HPP__
