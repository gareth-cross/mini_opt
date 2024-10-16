// Copyright (c) 2024 Gareth Cross.
// I copy-pasta'd this from the wrenfold project.
#pragma once
#include <chrono>
#include <memory>
#include <string_view>

#ifdef MINI_OPT_TRACING
// Concatenate `a` with `b`.
#define MINI_OPT_CONCAT_(a, b) a##b
#define MINI_OPT_CONCAT(a, b) MINI_OPT_CONCAT_(a, b)

// Create a scoped trace with the provided name.
// This version of the macro accepts a string literal.
#define MINI_OPT_SCOPED_TRACE_STR(str) \
  mini_opt::scoped_trace MINI_OPT_CONCAT(__timer, __LINE__) { str }
#else
// Do nothing when tracing is disabled.
#define MINI_OPT_SCOPED_TRACE_STR(str)
#endif  // MINI_OPT_ENABLE_TRACING

// Declare a named trace. `name` must be a string literal.
#define MINI_OPT_SCOPED_TRACE(name) MINI_OPT_SCOPED_TRACE_STR(#name)
#define MINI_OPT_FUNCTION_TRACE() MINI_OPT_SCOPED_TRACE_STR(__FUNCTION__)

#ifdef MINI_OPT_TRACING

namespace mini_opt {

// Fields are documented in this doc - I only support the bare minimum required:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/
struct trace_event {
  // Name of the event.
  std::string_view name;
  // Timestamp in microseconds.
  std::int64_t ts;
  // Process ID.
  std::uint32_t pid;
  // Thread ID.
  std::uint32_t tid;
  // Duration in nanoseconds.
  std::int64_t dur_ns;
};

// Aggregates tracing events and writes them out in chrome://trace format.
class trace_collector {
 public:
  trace_collector();

  // Access global instance of the trace collector.
  static trace_collector* get_instance();

  // Log an event.
  void submit_event(trace_event event);

  // Get all the traces as a JSON blob.
  std::string get_trace_json();

 private:
  struct trace_collector_impl& impl() noexcept;

  // Minimize header bloat with pimpl pattern.
  std::unique_ptr<trace_collector_impl> impl_;
};

// Measure time elapsed in a particular scope.
class scoped_trace {
 public:
  // Note we do not own `name`. It is assumed to be a global string literal.
  explicit scoped_trace(const std::string_view name) noexcept
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
  ~scoped_trace();

  // non-copyable and non-moveable
  scoped_trace(const scoped_trace&) = delete;
  scoped_trace(scoped_trace&&) = delete;
  scoped_trace& operator=(const scoped_trace&) = delete;
  scoped_trace& operator=(scoped_trace&&) = delete;

 private:
  std::string_view name_;
  std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace mini_opt

#endif  // MINI_OPT_TRACING
