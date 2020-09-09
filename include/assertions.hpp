#pragma once
#define ASSERTS_ENABLED

// Geneates an exception w/ a formatted string.
void RaiseAssert(const char* const condition, const char* const file, const int line,
                 const char* const fmt = 0, ...);

// Assertion macros.
// Based on: http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert
#ifdef ASSERTS_ENABLED
#define ASSERT_IMPL(cond, file, line, ...)           \
  do {                                               \
    if (!static_cast<bool>(cond)) {                  \
      RaiseAssert(#cond, file, line, ##__VA_ARGS__); \
    }                                                \
  } while (false)
#else
#define ASSERT_IMPL(cond, file, line, ...) \
  do {                                     \
    (void)sizeof((condition));             \
  } while (0)
#endif

// Macro to use when defining an assertion.
#define ASSERT(cond, ...) ASSERT_IMPL(cond, __FILE__, __LINE__, ##__VA_ARGS__)
