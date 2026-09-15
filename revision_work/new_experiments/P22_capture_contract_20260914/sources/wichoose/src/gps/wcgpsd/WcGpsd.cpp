/**
 * Class that implements a GPS daemon using fake, preset, data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <fcntl.h>  // open(), O_RDONLY, etc
#include <unistd.h> // read()
#include <cstdio>   // sscanf()
#include <cstdlib>  // strtof(), atoi()
#include <cstring>  // memset(), strcmp()
#include <string>   // std::string
#include <iostream> // std::cerr, std::endl

#include "../GpsInfo.hpp"                   // struct GpsInfo
#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/time/TimeUtil.hpp"     // class TimeUtil
#include "../../util/thread/Runner.hpp"     // class Runner
#include "../../util/log/LogFile.hpp"       // LOG_*

#include "WcGpsd.hpp" // class WcGpsd

#define LOG_FNAME "/var/log/wcgpsd.log"

#define SERIAL_DEVICE_DEF "/dev/ttyACM0"
#define HEAD_INIT_DEF 0 // degrees from north
#define HEAD_INIT_MIN 0
#define HEAD_INIT_MAX 359.9999999
#define HEAD_UPD_MIN_SPEED_DEF 0 // Km/h
#define HEAD_UPD_MIN_SPEED_MIN 0
#define HEAD_UPD_MIN_SPEED_MAX 100
#define POS_UPD_MIN_SPEED_DEF 0 // Km/h
#define POS_UPD_MIN_SPEED_MIN 0
#define POS_UPD_MIN_SPEED_MAX 100

#define LAT_INV 9999 // lat should be in [-90,90]
#define LON_INV 9999 // lat should be in [-180,180]

/**
 * Empty constructor.
 */
WcGpsd::WcGpsd() : GpsDaemon() {

  WcConfigFile configFile; // calls default constructor
  
  const std::string section("wcgpsd");

  // read serial device
  std::string serialDevice = configFile.strValue(section, 
                                                 "serial-device",
                                                 SERIAL_DEVICE_DEF);

  if ((this->serialFd = open(serialDevice.c_str(), O_RDONLY | O_NOCTTY)) < 0)
    LOG_FATAL_PERROR_EXIT("WcGpsd open() serial device");

  // read initial heading
  float headInit = configFile.floatValue(section,
                                         "head-init",
                                         HEAD_INIT_DEF,
                                         HEAD_INIT_MIN, 
                                         HEAD_INIT_MAX);
  
  // set default values
  this->curGpsInfo.head = headInit;
  this->curGpsInfo.lat = LAT_INV;
  this->curGpsInfo.lon = LON_INV;

  // read heading and position min speed update thresholds
  this->headUpdMinSpeed = configFile.floatValue(section,
                                                "head-upd-min-speed",
                                                HEAD_UPD_MIN_SPEED_DEF,
                                                HEAD_UPD_MIN_SPEED_MIN,
                                                HEAD_UPD_MIN_SPEED_MAX);

  this->posUpdMinSpeed = configFile.floatValue(section, 
                                               "pos-upd-min-speed",
                                               POS_UPD_MIN_SPEED_DEF,
                                               POS_UPD_MIN_SPEED_MIN,
                                               POS_UPD_MIN_SPEED_MAX);
}

/**
 * Updates curGpsInfo field accord to the latest information. Everything else is constant.
 * Overrides same-name method from superclass.
 */
void WcGpsd::updateGpsInfo() {
 
  // save last-known heading and position
  const float prevHead = this->curGpsInfo.head;
  const float prevLat = this->curGpsInfo.lat;
  const float prevLon = this->curGpsInfo.lon;
  
  memset(&this->curGpsInfo, 0, sizeof(GpsInfo)); // clean slate

  // loop variables
  char timestr[20];
  struct tm tm;
  char darray[NMEA_MAX_WORDS][NMEA_WORD_SIZE];
  NmeaType nmeaType = NmeaType::OTHER;
  bool wrmc = true, wgga = true, wgsa = true; // message waiting flags
  
  while (wrmc || wgga || wgsa) {
    if (!this->readNmea(&nmeaType, darray)) { // checksum failed
      LOG_WARN("read nmea checksum failed");
      continue;
    }

    switch (nmeaType) {
      case NmeaType::RMC:
        /* rmc has time and date:
           time is in position 1 and date in position 9 */
        snprintf(timestr, 20, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c",
                 darray[9][4], darray[9][5], darray[9][2],
                 darray[9][3], darray[9][0], darray[9][1],
                 darray[1][0], darray[1][1], darray[1][2],
                 darray[1][3], darray[1][4], darray[1][5]);

        if (strptime(timestr, "%Y-%m-%d %H:%M:%S", &tm) != 0)
          this->curGpsInfo.gpstime = mktime(&tm);
        else this->curGpsInfo.gpstime = 0;
          
        this->curGpsInfo.systime = TimeUtil::getSystimeMillis(); // set systime
        
        if (darray[2][0] == 'A') { // two options: 'A' (active) and 'V' (void)

          // handle latitude
          if (darray[3][0] != '\0' && darray[4][0] != '\0') {
            float lat1 = 0.0, lat2 = 0.0;
            sscanf(darray[3], "%2f%f", &lat1, &lat2);

            if (strcmp(darray[4], "N") == 0)
              this->curGpsInfo.lat = (lat1 + lat2 / 60.0);
            else if (strcmp(darray[4], "S") == 0)
              this->curGpsInfo.lat = -(lat1 + lat2 / 60.0);
          }

          // handle longitude
          if (darray[5][0] != '\0' && darray[6][0] != '\0') {
            float lon1 = 0.0, lon2 = 0.0;
            sscanf(darray[5], "%3f%f", &lon1, &lon2);

            if (strcmp(darray[6], "E") == 0)
              this->curGpsInfo.lon = (lon1 + lon2 / 60.0);
            else if (strcmp(darray[6], "W") == 0)
              this->curGpsInfo.lon = -(lon1 + lon2 / 60.0);
          }

          if (darray[7][0] != '\0') // 1.852 factor -> knot to km/h
            this->curGpsInfo.speed = strtof(darray[7], NULL) * 1.852;

          if (darray[8][0] != '\0')
            this->curGpsInfo.head = strtof(darray[8], NULL);
        }

        wrmc = false;
        break;

      case NmeaType::GGA:
        if (darray[7][0] != '\0')
          this->curGpsInfo.nsats = (uint8_t) atoi(darray[7]);
        
        if (darray[8][0] != '\0')
          this->curGpsInfo.hdop = strtof(darray[8], NULL);

        if (darray[9][0] != '\0')
          this->curGpsInfo.alt = strtof(darray[9], NULL);
      
        wgga = false;
        break;

      case NmeaType::GSA:
        if (darray[2][0] != '\0')
          // 1=nofix, 2=2D, 3=3D
          this->curGpsInfo.fix = (uint8_t) atoi(darray[2]);
        wgsa = false;
        break;

      default:
        break;
    } // switch (nmeaType) end
  } // while (wrmc || wgga || wgsa) end
  
  // replace heading if insufficient speed
  if (this->curGpsInfo.speed < this->headUpdMinSpeed)
    this->curGpsInfo.head = prevHead;
  
  // replace pos if insufficient speed and not first received position
  if (this->curGpsInfo.speed < this->posUpdMinSpeed && prevLat != LAT_INV) {
    this->curGpsInfo.lat = prevLat;
    this->curGpsInfo.lon = prevLon;
  }
  
}

/**
 * Reads  a NMEA sentence from the serial file descriptor into a string array that is easy to process.
 *
 * @param nmeaType: pointer to variable where type of sentence read is to be stored (output argument).
 * @param darray: pointer to array where NMEA data is to be stored (output argument).
 * @return true on success, false on checksum fail.
 */
bool WcGpsd::readNmea(NmeaType *nmeaType, char darray[][NMEA_WORD_SIZE]) {

  /* squeaky clean */
  memset(darray, '\0', NMEA_MAX_WORDS * NMEA_WORD_SIZE); // squeaky clean
  
  unsigned int checksum = 0;
  int ret = 0, nread = 0;
  char readbuf[NMEA_MAX_BUFLEN];
  char ch;
  char* bufptr = readbuf;
  while (1) { // hopefully not an infinite loop

    ret = read(this->serialFd, &ch, 1); // read something

    if (ret == -1){
      LOG_FATAL_PERROR_EXIT("main loop (reading from serial port)");
    } else if (ret == 0) { // EOF
      if (nread == 0) return 1; // no bytes read; return ERROR
      else break; // some bytes read; go on
    } else { // 'ret' must == 1 if we get here
      if (ch == '\n') break;
      if (ch == '\r') continue;

      if (nread < NMEA_MAX_BUFLEN-2) { // discard > (n - 1) bytes
        nread++;
        *bufptr++ = ch;
      } else break;
    }
  }
  
  // process what we've read
  readbuf[nread] = '\0';
  
  // figure out nmea message type by looking at 1st few chars
  *nmeaType = NmeaType::OTHER;
  if (nread > 6) { // discard too-short messages
    bufptr = readbuf + 3; // + 3 to skip the initial $GP (GPS) or $GN (GNSS)
    if (!strncmp(bufptr, "RMC", 3))
      *nmeaType = NmeaType::RMC;
    else if (!strncmp(bufptr, "GGA", 3))
      *nmeaType = NmeaType::GGA;
    else if (!strncmp(bufptr, "GSA", 3))
      *nmeaType = NmeaType::GSA;
  }

  if (*nmeaType != NmeaType::OTHER) { // do we have something to work with?
    bufptr = readbuf;
    int nwords = 0, i = 0, ret = 0;

    // %n gives us the index where the match ended
    while (nwords < NMEA_MAX_WORDS
        && (ret = sscanf(bufptr, "%[^,*]%n", darray[nwords], &i)) != EOF) {
      if (ret == 0) { // found a ',' or '*'
        i = 0;
        darray[nwords][0] = '\0'; // overkill because squeaky clean
      }

      nwords++;
      bufptr += i;

      /* should run everytime except for the last word (checksum).
        just to avoid consuming the terminating char */
      if (*bufptr == ',' || *bufptr == '*') bufptr++;
    }

    // checksum validation
    sscanf(darray[nwords - 1], "%X", &checksum);
    bufptr = readbuf + 1; // '$' shouldn't be used in checksum
    while (*bufptr != '*' && *bufptr != '\0') checksum ^= *bufptr++;
  }
  
  return checksum == 0; // 0 means good checksum
}

// WcGpsd class implementation end

/**
 * The procedure that actually bootstraps the program.
 *
 * @param argc: number of command line arguments, including the program's name.
 * @param argv: array of command line arguments, as strings.
 * @return 0 on success, other value on error.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wcgpsd");

  LOG_MSG("wcgpsd starting");
  
  if (argc > 1)
    std::cerr << "Ignoring all command-line arguments." << std::endl;

  WcGpsd gpsd; // create runnable
  
  Runner& runner = Runner::getInstance(); // get runner
  runner.addRunnable(gpsd);              // add runnable to runner
  runner.run();                          // actually run the runnable

  LOG_CLOSE();

  return 0;
}
