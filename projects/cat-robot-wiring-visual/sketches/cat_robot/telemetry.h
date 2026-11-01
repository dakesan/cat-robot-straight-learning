#pragma once
#include "drive.h"

namespace robot {
struct Sample {
  uint32_t sample = 0, timeMs = 0, epoch = 0, commandSequence = 0;
  int64_t encoderLeft = 0, encoderRight = 0;
  uint32_t invalidLeft = 0, invalidRight = 0;
  int left = 0, right = 0, targetLeft = 0, targetRight = 0, batteryMv = 0;
  float speed = 0, correction = 0, measuredLeft = 0, measuredRight = 0;
  bool armed = false, syncLed = false;
  DriveMode mode = DriveMode::Pwm;
  StopReason reason = StopReason::Startup;
};

template <unsigned Capacity = 256> struct Telemetry {
  Sample samples[Capacity];
  uint32_t sequence = 0;
  void push(Sample value) {
    value.sample = ++sequence;
    samples[(sequence - 1) % Capacity] = value;
  }
  unsigned read(uint32_t after, Sample* output, unsigned limit, uint32_t& dropped) const {
    const uint32_t first = sequence > Capacity ? sequence - Capacity + 1 : 1;
    uint32_t start = after + 1;
    dropped = start < first ? first - start : 0;
    if (start < first) start = first;
    unsigned count = 0;
    for (uint32_t current = start; current <= sequence && count < limit; ++current)
      output[count++] = samples[(current - 1) % Capacity];
    return count;
  }
};

inline bool syncLight(uint32_t elapsed) {
  return elapsed < 150 || (elapsed >= 500 && elapsed < 800) ||
         (elapsed >= 1500 && elapsed < 1950);
}
}  // namespace robot
