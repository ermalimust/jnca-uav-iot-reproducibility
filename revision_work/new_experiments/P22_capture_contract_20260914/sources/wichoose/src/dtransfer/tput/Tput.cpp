/**
 * Implelents methods for structures  useful for representing throughput samples.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "Tput.hpp" // struct TputSamp

/**
 * Constructor method that creates an empty tput vector with the contents provided as an argument.
 *
 * @param tputVec: desired initial size for byteCount vector.
 * @param source: source for the values in this sample.
 */
TputSamp::TputSamp(TputVec const&  tputVec,
                   const TputSource source) :
                   tputVec(tputVec), source(source) {}

/**
 * Constructor method that creates an empty tput vector of a given size..
 *
 * @param ncounts: desired initial size for byteCount vector.
 * @param source: source for the values in this sample.
 */
TputSamp::TputSamp(const std::size_t ncounts,
                   const TputSource source) :
                   tputVec(ncounts), source(source) {}
