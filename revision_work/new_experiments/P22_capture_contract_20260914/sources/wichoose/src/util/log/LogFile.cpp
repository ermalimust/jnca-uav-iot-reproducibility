/**
 * Implementation of LogFile utility class, used to log program errors.
 *
 * Rui Meireles, Spring 2021 (@vassar.edu)
 */

#include <sys/stat.h> // stat()
#include <cstring>    // std::strerror
#include <cerrno>     // errno
#include <sstream>    // std::stringstream

#include "../conf/WcConfigFile.hpp" // class WcConfigFile

#include "LogFile.hpp" // class LogFile

LogFile* LogFile::singleton = nullptr; // initialize the singleton

/**
 * Empty protected constructor.
 */
LogFile::LogFile() : logfp(nullptr), level(LOG_LEVEL_DEF) { }

/**
 * Returns the singleton LogFile instance, creating it in the process if needed.
 *
 * @return pointer to LogFile instance.
 */
LogFile* LogFile::getInstance() {
  if (LogFile::singleton == nullptr) LogFile::singleton = new LogFile();
  return LogFile::singleton;
}

/**
 * Initialize the log file using the provided filename.
 *
 * @param fname: name of the file to use for the log.
 */
void LogFile::initLog(const char* fname) {
  this->initLog(fname, LOG_RESET_LEN_THRES_DEF);
}

/**
 * Initialize the log file using the provided filename. Additionally, set the log level according to the
 * value provided in the configuration file for program named pname.
 *
 * @param fname: name of the file to use for the log.
 * @param pname: name of the program to use to set the log level from the config file.
 */
void LogFile::initLog(const char* fname, const char* pname) {
  this->initLog(fname, LOG_RESET_LEN_THRES_DEF);
  
  WcConfigFile configFile; // calls default constructor
  
  // read and set log level
  LogLevel logLevel = configFile.logLevel(pname, LOG_LEVEL_DEF);
  this->setLevel(logLevel);
}

/**
 * Initialize the log using the provided filename. Reset the log file if it is larger than the provided threshold.
 *
 * @param fname: name of the file to use for the log.
 * @param restLenThres: size threshold beyond which the log file is to be reset.
 */
void LogFile::initLog(const char* fname, const std::size_t resetLenThres) {

  if (fname != nullptr) {
    /* check the size */
    struct stat fstat;
    stat(fname, &fstat);

    /* reset log if it's larger than limit */
    if ((size_t)fstat.st_size > resetLenThres)
      this->logfp = fopen(fname, "w");
    else this->logfp = fopen(fname, "a+");

    if (this->logfp != nullptr)
      fseek(this->logfp, 0, SEEK_END); /* scan to the end */
  }
}

/**
 * Configures the minimum severity level for messages to be written to the log.
 *
 * @param level: the minimum severity level.
 */
void LogFile::setLevel(const LogLevel level) { this->level = level; }

/**
 * Writes a message to the underlying log file.
 *
 * @param level: severity of the message.
 * @param msg: the message to be written.
 * @param sfname: the name of the source file the message comes from.
 * @param slineno: the number of the source file line the message comes from.
 */
void LogFile::writeLog(LogLevel level, const char* msg, const char* sfname,
                       int slineno) {

  if (this->logfp != nullptr && level <= this->level) {
    static const char* levelNames[] = {"fatal", "error", "warn", "msg",
                                       "verbose"};
    const char*  ptype = levelNames[level];
    
    time_t rawtime;
    struct tm *ptm;

    time(&rawtime);
    ptm = gmtime(&rawtime);

    fprintf(logfp, "%s\t%4d-%02d-%02d\t%02d:%02d:%02d\t%s\t%d\t%s\n", ptype,
        1900 + ptm->tm_year, 1 + ptm->tm_mon, ptm->tm_mday,
        ptm->tm_hour, ptm->tm_min, ptm->tm_sec, sfname, slineno, msg);
    fflush(logfp);
  }
}

/**
 * Writes a message to the underlying log file, along with the error associated with the global errno,
 * which is assumed to have been previously set.
 *
 * @param level: severity of the message.
 * @param msg: the message to be written.
 * @param sfname: the name of the source file the message comes from.
 * @param slineno: the number of the source file line the message comes from.
 */
void LogFile::writeLogPerror(LogLevel level, const char* msg,
                             const char* sfname, int slineno) {
  std::stringstream ss;
  ss << msg << ": " << std::strerror(errno);
  this->writeLog(level, ss.str().c_str(), sfname, slineno);
}

/**
 * Closes the underlying log file. Nothing will be written after a call to this method until initLog() is
 * called again.
 */
void LogFile::closeLog() {
  if (this->logfp != nullptr) {
    fclose(this->logfp);
    this->logfp = nullptr;
  }
}
