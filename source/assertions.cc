// Copyright 2020 Gareth Cross
#include "mini_opt/assertions.hpp"

#include <stdarg.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>

void RaiseAssert(const char* const condition, const char* const file, const int line,
                 const char* const fmt, ...) {
  // TODO(gareth): Just use libfmt instead of gross varargs?
  char string_buffer[8192];
  memset(string_buffer, 0, sizeof(string_buffer));
  if (fmt) {
    va_list args;
    va_start(args, fmt);
#ifndef _MSC_VER
    vsprintf(string_buffer, fmt, args);
#else
    vsprintf_s(string_buffer, sizeof(string_buffer), fmt, args);
#endif
    va_end(args);
  }
  std::stringstream err;
  err << "Assertion failed: " << condition;
  err << "\nFile: " << file << " (line " << std::to_string(line) << ")";
  if (strlen(string_buffer) > 0) {
    err << "\nReason: " << std::string(string_buffer);
  }
  throw std::runtime_error(err.str());
}
