#!/bin/sh

##########################################################
##  Script used to control wichoose execution.          ##
##  Rui Meireles (@vassar.edu) 2021, 2024               ##
##########################################################

# return: a timestamp formated in a human-readable format
timestamp() {
  date +"%Y.%m.%d-%H.%M.%S"
}

# launches a single process, passed in as an argument
# $1: path to executable to be launched
# $2: string containing arguments to pass to the executable
# $3: name of file to redirect process output to (optional)
launch() {

  # rename arguments for convenience
  program=$1
  args=$2
  logSuffix=$3
  
  echo "  Launching $program..."
 
  # just in case it was already running
  killall -q $program
  sleep 1
    
  if [ -z "$logSuffix" ]; then # wo/ output redirection
    nohup $program $args > /dev/null 2>&1 &
  else                         # w/ output redirection
    nohup $program $args  > ./log/$program-$logSuffix.csv 2>&1 &
  fi

  sleep 1
}

# starts all processes for receiver or transmitter, depending on argument
# param $1: rx for receiver, tx for transmitter
start() {
  tstamp=$(timestamp)
  echo "Starting wichoose ($tstamp)..."
  
  mkdir -p log # make sure there is a log directory
  
  if [ "$1" = "rx" ]; then
    launch "./wcgpsd" ""
    launch "./wcgpsprinter" "" $tstamp
    launch "./wcreceiver" "" $tstamp
      
  elif [ "$1" = "tx" ]; then
    launch "./wcgpsd" ""
    launch "./wcgpsprinter" "" $tstamp
    launch "./wcrssiprinter" "" $tstamp
    launch "./wichoicemaker" "--print-tput-est" $tstamp
    launch "./wichoiceprinter" "" $tstamp
    launch "./wcsender" "" $tstamp

  else
    echo "Must specify one of (rx, tx) for start"
  fi
  echo "Done!"
}

# stops all possibly-running processes
stop() {

  tstamp=$(timestamp)
  echo "Stopping wichoose ($tstamp)..."

  # stop all possible processes because we don't know whether the receiver
  # or transmitter is running
  programs="wichoiceprinter wichoicemaker wcreceiver wcsender wcrssiprinter wcgpsprinter wcgpsd wcgpsdfaker"
 
  for program in $programs
  do
    echo "  Killing $program..."
    killall -q $program
    sleep 1
  done
  
  # now move the logs
  echo "  Moving /var/log logs..."

  mkdir -p log # make sure there is a log directory

  for program in $programs
  do
    # move file, but only if it exists (prevents error messages)
    if [ -f /var/log/$program.log ]; then
      mv /var/log/$program.log ./log/$program-$tstamp.log
    fi
  done
  
  echo "Done!"
}

# prints the names and ids of all running processes related to wichoose.
status() {
  echo "Running processes:"
  ps | grep "wichoi\|wc" | grep -v "grep" # -v inverts results so grep is excluded
}

# the main function
# params:
# $1: start, stop, or status
# $2: if $1 == start, rx or tx, otherwise ignored
main() {
  if [ "$1" = "start" ]; then
    start "$2"
  elif [ "$1" = "stop" ]; then
    stop
  elif [ "$1" = "status" ]; then
    status
  else
    echo "Must specify argument: start (rx or tx), stop, or status"
  fi
}

# bootstrap program by calling main
main $@
