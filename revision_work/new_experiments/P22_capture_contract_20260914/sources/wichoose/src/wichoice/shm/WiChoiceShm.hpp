/**
 * Defines a data structure for an interface choice shared memory.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_SHM_HPP__
#define WI_CHOICE_SHM_HPP__

#include "../../util/shm/Shm.hpp" // struct Shm
#include "../WiChoice.hpp"        // struct WiChoice

struct WiChoiceShm : Shm {
  WiChoice wiChoiceMes; // choice made from measured throughput
  WiChoice wiChoiceEst; // choice made from estimated throughput
};
 
#endif // WI_CHOICE_SHM_HPP__
