/**
 * Defines data structure to represent a network.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NETWORK_HPP__
#define NETWORK_HPP__

#include <string>  // std::string
#include <cstdint> // uint*_t
#include <map>     // std::map

#include "WifiStandard.hpp" // enum WifiStandard

typedef uint8_t NetworkId; // to prevent mistakes

struct Network {
  NetworkId id;  // should be unique
  std::string iface;
  std::string ipCli;
  std::string ipSrv;
  WifiStandard wiStandard;
};

typedef std::map<NetworkId, Network> NetworkMap; // net id -> network

#endif // NETWORK_HPP__
