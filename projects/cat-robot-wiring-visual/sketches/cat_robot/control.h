#pragma once

#include <stdint.h>
#include <stdlib.h>

namespace robot {
constexpr uint32_t COMMAND_TIMEOUT_MS = 300;
constexpr uint32_t ENCODER_TIMEOUT_MS = 800;
constexpr int MAX_PERCENT = 50;
constexpr int SERVO_MIN_US = 1400;
constexpr int SERVO_MAX_US = 1600;

enum class StopReason {
  Startup, Operator, CommandTimeout, EncoderLeft, EncoderRight,
  BatteryLow, BatterySensor, InvalidCommand, OutputError, NetworkError
};

inline const char* reasonName(StopReason reason) {
  switch (reason) {
    case StopReason::Startup: return "startup";
    case StopReason::Operator: return "operator";
    case StopReason::CommandTimeout: return "command_timeout";
    case StopReason::EncoderLeft: return "encoder_left_no_response";
    case StopReason::EncoderRight: return "encoder_right_no_response";
    case StopReason::BatteryLow: return "battery_low";
    case StopReason::BatterySensor: return "battery_sensor";
    case StopReason::InvalidCommand: return "invalid_command";
    case StopReason::OutputError: return "output_error";
    case StopReason::NetworkError: return "network_error";
  }
  return "unknown";
}

inline int8_t encoderDelta(uint8_t previous, uint8_t current) {
  const uint8_t changed = previous ^ current;
  if (changed == 0 || changed == 3) return 0;
  return ((previous >> 1) ^ (current & 1)) ? -1 : 1;
}

struct Controller {
  bool armed = false;
  bool invertLeft = false;
  bool invertRight = false;
  bool batteryEnabled = false;
  uint32_t epoch = 0;
  uint32_t sequence = 0;
  uint32_t lastCommand = 0;
  int targetLeft = 0, targetRight = 0, targetServo = 0;
  int left = 0, right = 0, servo = 0;
  StopReason reason = StopReason::Startup;
  int64_t previousLeft = 0, previousRight = 0;
  uint32_t leftProgressAt = 0, rightProgressAt = 0;
  bool lowBattery = false;
  uint32_t lowBatteryAt = 0;

  void stop(StopReason why) {
    armed = false;
    targetLeft = targetRight = targetServo = 0;
    left = right = servo = 0;
    reason = why;
  }

  bool arm(uint32_t now, int batteryMv) {
    if (batteryEnabled && (batteryMv < 5500 || batteryMv > 9000)) {
      stop(StopReason::BatterySensor);
      return false;
    }
    if (batteryEnabled && batteryMv < 6800) {
      stop(StopReason::BatteryLow);
      return false;
    }
    stop(StopReason::Operator);
    ++epoch;
    if (epoch == 0) ++epoch;
    sequence = 0;
    armed = true;
    lastCommand = leftProgressAt = rightProgressAt = now;
    lowBattery = false;
    return true;
  }

  bool command(uint32_t now, uint32_t commandEpoch, uint32_t seq,
               int newLeft, int newRight, int newServo) {
    if (!armed || commandEpoch != epoch || seq <= sequence) return false;
    if (static_cast<uint32_t>(now - lastCommand) >= COMMAND_TIMEOUT_MS) {
      stop(StopReason::CommandTimeout);
      return false;
    }
    if (newLeft < -MAX_PERCENT || newLeft > MAX_PERCENT ||
        newRight < -MAX_PERCENT || newRight > MAX_PERCENT ||
        (newServo != 0 && (newServo < SERVO_MIN_US || newServo > SERVO_MAX_US))) {
      stop(StopReason::InvalidCommand);
      return false;
    }
    sequence = seq;
    lastCommand = now;
    targetLeft = newLeft;
    targetRight = newRight;
    targetServo = newServo;
    return true;
  }

  static int ramp(int current, int target) {
    if ((current > 0 && target < 0) || (current < 0 && target > 0)) target = 0;
    if (current < target) return current + (target - current > 2 ? 2 : target - current);
    if (current > target) return current - (current - target > 2 ? 2 : current - target);
    return current;
  }

  void tick(uint32_t now, int64_t leftCount, int64_t rightCount, int batteryMv) {
    if (!armed) return;
    if (static_cast<uint32_t>(now - lastCommand) >= COMMAND_TIMEOUT_MS) {
      stop(StopReason::CommandTimeout);
      return;
    }
    if (batteryEnabled) {
      if (batteryMv < 5500 || batteryMv > 9000) {
        stop(StopReason::BatterySensor);
        return;
      }
      if (batteryMv < 6800) {
        if (!lowBattery) { lowBattery = true; lowBatteryAt = now; }
        if (static_cast<uint32_t>(now - lowBatteryAt) >= 200) {
          stop(StopReason::BatteryLow);
          return;
        }
      } else lowBattery = false;
    }
    left = targetLeft == 0 ? 0 : ramp(left, targetLeft);
    right = targetRight == 0 ? 0 : ramp(right, targetRight);
    if (targetServo == 0) servo = 0;
    else {
      if (servo == 0) servo = 1500;
      const int difference = targetServo - servo;
      servo += difference > 10 ? 10 : difference < -10 ? -10 : difference;
    }
    if (abs(left) < 15 || leftCount != previousLeft) leftProgressAt = now;
    if (abs(right) < 15 || rightCount != previousRight) rightProgressAt = now;
    previousLeft = leftCount;
    previousRight = rightCount;
    if (static_cast<uint32_t>(now - leftProgressAt) >= ENCODER_TIMEOUT_MS)
      stop(StopReason::EncoderLeft);
    else if (static_cast<uint32_t>(now - rightProgressAt) >= ENCODER_TIMEOUT_MS)
      stop(StopReason::EncoderRight);
  }
};
}  // namespace robot
