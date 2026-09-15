/**
 * Defines a class to send data in bulk across all available wifi interfaces.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef DATA_SENDER_ALL_H__
#define DATA_SENDER_ALL_H__

#include "DataSender.hpp" // class DataSender

class DataSenderAll : public DataSender {

public:
  /**
   * Constructor method.
   */
  explicit DataSenderAll(); // explicit prevents implicit typecast

protected:
  /**
   * Implements threads that continually send data at the fastest possible rate over all wifi interfaces.
   */
  void commThread() override;
};

#endif // DATA_SENDER_ALL_H__
