#!/bin/sh

################################################################
##  Script used to copy logs from routers after termination.  ##
##  Rui Meireles (@vassar.edu) 2024               ##
################################################################

# global variables
iprx="10.227.16.239"
iptx="10.227.16.241"
srcfolder="~/log/*"

# return: a timestamp formated in a human-readable format
timestamp() {
  date +"%Y.%m.%d-%H.%M.%S"
}

# copies the logs from node passed in as argument
# $1: node to copy logs from, rx or tx
copy() {
  node=$1 # node is first argument
  ip=""

  if [ "$node" = "rx" ]; then
    ip=$iprx
  elif [ "$node" = "tx" ]; then
    ip=$iptx
  else
    echo "Must specify one of (rx, tx) for copy operation"
    return
  fi
  
  tstamp=$(timestamp)
  dstfolder=logs-$node-$tstamp

  mkdir -p $dstfolder # make sure the target directory exists
  
  echo "Copying from $ip:$srcfolder to $dstfolder..."

  scp -r -oHostKeyAlgorithms=+ssh-rsa -O root@$ip:$srcfolder $dstfolder
  
  echo "Done!"
}

# bootstrap program by calling copy function
copy $@
