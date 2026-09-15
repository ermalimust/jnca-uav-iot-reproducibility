/**
 * Defines structures and types useful for representing throughput samples (either measured or estimated).
 * Separated onto its own header file to ensure a consistent definition across the many places it is used.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef TPUT_HPP__
#define TPUT_HPP__

#include <cstdint> // uint*_t
#include <cstddef> // std::size_t
#include <vector>  // std::vector
#include <map>     // std::map
#include <set>     // std::set

/**
 * A throughput value.
 */
typedef uint32_t Tput; // 32 bits good are enough for 34 gigabits

/**
 * A vector of throughput values.
 */
typedef std::vector<Tput> TputVec;

/**
 * Enumerates the possible sources for throughput samples (estimation and measurement).
 */
enum TputSource { ESTIMATE=0, MEASUREMENT=1, NSOURCES = 2};

typedef std::set<TputSource> TputSourceSet;

/**
 * A throughput sample is composed of a vector of observations (one per network), and a source.
 */
struct TputSamp {
  /**
   * Constructor method that creates an empty tput vector with the contents provided as an argument.
   *
   * @param tputVec: desired initial size for byteCount vector.
   * @param source: source for the values in this sample.
   */
  TputSamp(TputVec const& tputVec, const TputSource source);

  /**
   * Constructor method that creates an empty tput vector of a given size..
   *
   * @param ncounts: desired initial size for byteCount vector.
   * @param source: source for the values in this sample.
   */
  TputSamp(const std::size_t ncounts, const TputSource source);

  TputVec tputVec;
  const TputSource source;
};

/**
 * A vector of throughput samples.
 */
typedef std::vector<TputSamp> TputSampVec;

/**
 * A map of throughput samples where the key is the gpstime at which the samples where collected.
 */
typedef std::map<uint32_t, TputSamp> TputSampMap;

#endif // TPUT_HPP__
