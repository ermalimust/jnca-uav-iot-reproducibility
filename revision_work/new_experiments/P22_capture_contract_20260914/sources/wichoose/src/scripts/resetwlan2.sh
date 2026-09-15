#!/bin/sh

##############################################################
##  Script used to reset the temperamental 11ad interface.  ##
##  Rui Meireles (@vassar.edu) 2024             ##
##############################################################

ifconfig wlan2 down; ifconfig wlan2 up
