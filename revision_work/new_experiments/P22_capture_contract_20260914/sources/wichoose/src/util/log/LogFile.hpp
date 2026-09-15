/**
 * Utility class to log program errors.
 *
 * Rui Meireles, Spring 2021 (@vassar.edu)
 */

#ifndef LOG_FILE_HPP__
#define LOG_FILE_HPP__

#include <cstddef>  // std::size_t
#include <cstdlib>  // std::exit()
#include <iostream> // std::cerr, std::endl

// some default constants
#define LOG_LEVEL_DEF ERROR
#define LOG_RESET_LEN_THRES_DEF 1048576 // 1 MiB

// some useful macros
#define LOG_INIT(fname, pname) do{if(fname!= NULL)LogFile::getInstance()->initLog(fname, pname); else std::cerr << "error init log file " << fname << std::endl;}while(0);
#define LOG_SET_LEVEL(level) LogFile::getInstance()->setLevel(level);
#define LOG_VERBOSE(msg) LogFile::getInstance()->writeLog(VERBOSE, msg, __FILE__, __LINE__);
#define LOG_MSG(msg) LogFile::getInstance()->writeLog(MSG, msg, __FILE__, __LINE__);
#define LOG_WARN(msg) LogFile::getInstance()->writeLog(WARN, msg, __FILE__, __LINE__);
#define LOG_ERR(msg) LogFile::getInstance()->writeLog(ERROR, msg, __FILE__, __LINE__);
#define LOG_FATAL(msg) LogFile::getInstance()->writeLog(FATAL, msg, __FILE__, __LINE__);
#define LOG_FATAL_EXIT(msg) do{LOG_FATAL(msg) std::exit(1);}while(0);
#define LOG_FATAL_PERROR(msg) LogFile::getInstance()->writeLogPerror(FATAL, msg, __FILE__, __LINE__);
#define LOG_FATAL_PERROR_EXIT(msg) do{LOG_FATAL_PERROR(msg) std::exit(1);}while(0);
#define LOG_CLOSE() LogFile::getInstance()->closeLog();

// level of detail contained in log (increasing ordeR)
enum LogLevel {FATAL, ERROR, WARN, MSG, VERBOSE, NLOG_LEVELS};

class LogFile {

public:
  /**
   * Returns the singleton LogFile instance, creating it in the process if needed.
   *
   * @return pointer to LogFile instance.
   */
  static LogFile* getInstance();

  /**
   * Initialize the log file using the provided filename.
   *
   * @param fname: name of the file to use for the log.
   */
  void initLog(const char* fname);

  /**
   * Initialize the log file using the provided filename. Additionally, set the log level according to the
   * value provided in the configuration file for program named pname.
   *
   * @param fname: name of the file to use for the log.
   * @param pname: name of the program to use to set the log level from the config file.
   */
  void initLog(const char* fname, const char* pname);
  
  /**
   * Initialize the log using the provided filename. Reset the log file if it is larger than the provided threshold.
   *
   * @param fname: name of the file to use for the log.
   * @param restLenThres: size threshold beyond which the log file is to be reset.
   */
  void initLog(const char* fname, const std::size_t resetLenThres);

  /**
   * Configures the minimum severity level for messages to be written to the log.
   *
   * @param level: the minimum severity level.
   */
  void setLevel(const LogLevel level);

  /**
   * Writes a message to the underlying log file.
   *
   * @param level: severity of the message.
   * @param msg: the message to be written.
   * @param sfname: the name of the source file the message comes from.
   * @param slineno: the number of the source file line the message comes from.
   */
  void writeLog(LogLevel level, const char* msg, const char* sfname,
                int slineno);

  /**
   * Writes a message to the underlying log file, along with the error associated with the global errno,
   * which is assumed to have been previously set.
   *
   * @param level: severity of the message.
   * @param msg: the message to be written.
   * @param sfname: the name of the source file the message comes from.
   * @param slineno: the number of the source file line the message comes from.
   */
  void writeLogPerror(LogLevel level, const char* msg, const char* sfname,
                      int slineno);

  /**
   * Closes the underlying log file. Nothing will be written after a call to this method until initLog() is
   * called again.
   */
  void closeLog();

  /**
   * Singletons shouldn't be cloneable, so we eliminate the possibility.
   */
  LogFile(LogFile& other) = delete;

  /**
   * Singletons shouldn't be assignable.
   */
  void operator=(const LogFile&) = delete;
  
protected:
  static LogFile* singleton; // singleton instance

  FILE *logfp;
  LogLevel level;

  /**
   * Empty protected constructor.
   */
  LogFile();
};

#endif // LOG_FILE_HPP__
