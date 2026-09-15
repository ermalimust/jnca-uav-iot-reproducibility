/**
 * Defines generic circular iterator that can be applied to any STL container.
 *
 * Rui Meireles (@vassar.edu) 2025
 */

#ifndef CIRCULAR_ITERATOR_HPP__
#define CIRCULAR_ITERATOR_HPP__

#include <iterator> // std::iterator


template<typename Container>
class CircularIterator {

public:
  using Iterator = typename Container::iterator;
  using ValueType = typename Container::value_type;

  /**
   * Constructor method.
   *
   * @param container: the container to be traversed.
   */
  CircularIterator(Container& container);

  /**
   * Returns current element and advances iterator, going back to the first if at the end of the container.
   *
   * @return current element.
   */
  ValueType& next();
  
  /**
   * Moves iterator backwards, going back to the last element if at the start of the container, and then returns
   * the new current element.
   *
   * @return current element after the iterator is moved backwards..
   */
  ValueType& prev();
  
  /**
   * Returns current element withouth changing the  iterator.
   *
   * @return current element.
   */
  ValueType& current();
  
  /**
   * Resets the iterator to point to the start of the container.
   * Reset should be called if the underlying container has changed between iterator creation and usage.
   */
  void reset();

private:
  Container& container; // the container being traversed
  Iterator itr;         // the underlying simple iterator
};

#include "CircularIterator.tpp"  // implementation - mandatory for templates

#endif // CIRCULAR_ITERATOR_HPP__
