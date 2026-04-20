#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Sequences.hpp"

enum level_t : uint8_t { L_DEBUG, L_INFO, L_WARN, L_ERROR, L_FATAL };

namespace Logger {

inline level_t &minLevel() {
#ifdef DEBUG_MODE
  static level_t level = L_DEBUG;
#else
  static level_t level = L_INFO;
#endif
  return level;
}

inline void setLevel(level_t level) { minLevel() = level; }

inline std::string timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  std::ostringstream oss;
  oss << std::put_time(std::localtime(&time), "%H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return oss.str();
}

inline const char *levelTag(level_t level) {
  switch (level) {
  case L_DEBUG:
    return PURPLE "DBG" RESET;
  case L_INFO:
    return BLUE "INF" RESET;
  case L_WARN:
    return YELLOW "WRN" RESET;
  case L_ERROR:
    return RED "ERR" RESET;
  case L_FATAL:
    return BOLD RED "FTL" RESET;
  }
  return "???";
}

inline void print(level_t level, const char *file, int line, const char *msg) {
  if (level < minLevel())
    return;
  const char *filename = file;
  for (const char *p = file; *p; ++p)
    if (*p == '/')
      filename = p + 1;

  std::cout << BLUE << timestamp() << RESET " [" << levelTag(level) << "] "
            << PURPLE << filename << ":" << line << RESET BOLD " - " RESET
            << msg << std::endl;
}

inline void print(level_t level, const char *file, int line,
                  const std::string &msg) {
  print(level, file, line, msg.c_str());
}

} // namespace Logger

#define LOG_DEBUG(msg) Logger::print(L_DEBUG, __FILE__, __LINE__, msg)
#define LOG_INFO(msg) Logger::print(L_INFO, __FILE__, __LINE__, msg)
#define LOG_WARN(msg) Logger::print(L_WARN, __FILE__, __LINE__, msg)
#define LOG_ERROR(msg) Logger::print(L_ERROR, __FILE__, __LINE__, msg)
#define LOG_FATAL(msg) Logger::print(L_FATAL, __FILE__, __LINE__, msg)
