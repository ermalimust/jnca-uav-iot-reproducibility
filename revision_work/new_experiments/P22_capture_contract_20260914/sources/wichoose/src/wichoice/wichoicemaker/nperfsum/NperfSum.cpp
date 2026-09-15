/**
 * Implements structures that represent a network performance summary.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "NperfSum.hpp" // structs NperfSum, NperfSums

/**
 * Replaces the contents with contents from the argument object, as long as it has entries.
 *
 * @param src: the performance summary to copy from.
 */
void NperfSum::replaceWithIfNonZero(NperfSum const& src) {
  if (src.nentries > 0) {
    this->nentries = src.nentries;
    this->tputSum = src.tputSum;
  }
}

/**
 * Zeroes out the performance summary on which it is called.
 */
void NperfSum::reset() {
  this->nentries = 0;
  this->tputSum = 0;
}

/**
 * Replaces the contents with contents from the argument object, as long as it has entries.
 *
 * @param src: the performance summaries to copy from.
 */
void NperfSums::replaceWithIfNonZero(NperfSums const& src) {
  this->est.replaceWithIfNonZero(src.est);
  this->mes.replaceWithIfNonZero(src.mes);
}

/**
 * Zeroes out the performance summaries on which it is called.
 */
void NperfSums::reset() {
  this->est.reset();
  this->mes.reset();
}
