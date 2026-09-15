#!/bin/sh

##########################################################
##  Script used to set date to a given unix timestamp.  ##
##  Rui Meireles (@vassar.edu) 2024         ##
##########################################################

if [ -z "$1" ]; then
  echo "Must provide Unix timestamp argument."
else
  date +%s -s @$1
  echo -n "Current date: "; date
fi


