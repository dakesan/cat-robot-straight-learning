#pragma once

#include <math.h>
#include "control.h"

namespace robot {
inline uint32_t motorDuty(int percent) { return (abs(percent) * 255 + 50) / 100; }
enum class DriveMode { Pwm = 0, Speed = 1, Learned = 2, Student = 3 };

struct DriveCalibration {
  uint32_t version = 1;
  bool calibrated = false;
  float leftMetersPerCount = 0, rightMetersPerCount = 0;
  int leftSign = 1, rightSign = 1;
  float leftFeedforward = 0, rightFeedforward = 0;
  float leftDeadband = 0, rightDeadband = 0;
  float kp = 100, ki = 100, maxSpeed = 0;

  bool valid() const {
    return version == 1 && calibrated && isfinite(leftMetersPerCount) &&
      isfinite(rightMetersPerCount) && leftMetersPerCount > 0 && rightMetersPerCount > 0 &&
      (leftSign == 1 || leftSign == -1) && (rightSign == 1 || rightSign == -1) &&
      isfinite(leftFeedforward) && leftFeedforward > 0 &&
      isfinite(rightFeedforward) && rightFeedforward > 0 &&
      isfinite(leftDeadband) && leftDeadband >= 0 && leftDeadband < MAX_PERCENT &&
      isfinite(rightDeadband) && rightDeadband >= 0 && rightDeadband < MAX_PERCENT &&
      isfinite(kp) && kp >= 0 && isfinite(ki) && ki >= 0 &&
      isfinite(maxSpeed) && maxSpeed > 0 && maxSpeed <= 1;
  }
};

struct CorrectionModel {
  bool fitted = false, measured = false;
  float minSpeed = 0, maxSpeed = 0;
  float minCorrection = -0.2f, maxCorrection = 0.2f;
  float mean[4] = {}, scale[4] = {1, 1, 1, 1};
  float coefficients[2][4] = {};
  float intercept[2] = {};

  bool valid() const {
    if (!fitted || !isfinite(minSpeed) || !isfinite(maxSpeed) ||
        minSpeed <= 0 || maxSpeed < minSpeed || !isfinite(minCorrection) ||
        !isfinite(maxCorrection) || minCorrection < -0.2f || maxCorrection > 0.2f ||
        minCorrection > maxCorrection) return false;
    for (int i = 0; i < 4; ++i) {
      if (!isfinite(mean[i]) || !isfinite(scale[i]) || scale[i] <= 0) return false;
      for (int j = 0; j < 2; ++j) if (!isfinite(coefficients[j][i])) return false;
    }
    return isfinite(intercept[0]) && isfinite(intercept[1]);
  }

  void predict(float speed, float correction, float& lateral, float& heading) const {
    const float features[4] = {speed, correction, speed * correction, correction * correction};
    lateral = intercept[0]; heading = intercept[1];
    for (int i = 0; i < 4; ++i) {
      const float feature = (features[i] - mean[i]) / scale[i];
      lateral += coefficients[0][i] * feature;
      heading += coefficients[1][i] * feature;
    }
  }

  bool choose(float speed, float& correction) const {
    if (!valid() || !isfinite(speed) || speed < minSpeed || speed > maxSpeed) return false;
    float best = INFINITY;
    correction = 0;
    for (int step = -200; step <= 200; ++step) {
      const float candidate = step / 1000.0f;
      if (candidate < minCorrection - 1e-6f || candidate > maxCorrection + 1e-6f) continue;
      float lateral, heading;
      predict(speed, candidate, lateral, heading);
      const float loss = lateral * lateral / (0.05f * 0.05f) +
        heading * heading / (0.0872664626f * 0.0872664626f) + candidate * candidate * 1e-5f;
      if (loss < best) { best = loss; correction = candidate; }
    }
    return isfinite(best);
  }
};

// Small learned corrector run every 100 ms on top of the wheel-speed PI. Inputs are only
// what the ESP32 measures: target speed, time since the drive started (capped), filtered
// wheel speeds and the heading implied by the encoders. Output is the left/right
// correction, clamped to the trained range. It never renews the command lease.
struct StudentModel {
  static constexpr int INPUTS = 5, HIDDEN = 16;
  bool fitted = false, measured = false;
  float minSpeed = 0, maxSpeed = 0, minCorrection = -0.2f, maxCorrection = 0.2f;
  float trackM = 0.14f, elapsedCap = 3.0f;
  float mean[INPUTS] = {}, scale[INPUTS] = {1, 1, 1, 1, 1};
  float w1[HIDDEN][INPUTS] = {}, b1[HIDDEN] = {};
  float w2[HIDDEN][HIDDEN] = {}, b2[HIDDEN] = {};
  float w3[HIDDEN] = {}, b3 = 0;

  bool valid() const {
    if (!fitted || !isfinite(minSpeed) || !isfinite(maxSpeed) || minSpeed <= 0 ||
        maxSpeed < minSpeed || minCorrection < -0.2f || maxCorrection > 0.2f ||
        minCorrection > maxCorrection || !(trackM > 0) || !(elapsedCap > 0)) return false;
    for (int i = 0; i < INPUTS; ++i)
      if (!isfinite(mean[i]) || !isfinite(scale[i]) || scale[i] <= 0) return false;
    return true;
  }

  float predict(const float input[INPUTS]) const {
    float x[INPUTS], h1[HIDDEN], h2[HIDDEN];
    for (int i = 0; i < INPUTS; ++i) x[i] = (input[i] - mean[i]) / scale[i];
    for (int j = 0; j < HIDDEN; ++j) {
      float sum = b1[j];
      for (int i = 0; i < INPUTS; ++i) sum += w1[j][i] * x[i];
      h1[j] = tanhf(sum);
    }
    for (int j = 0; j < HIDDEN; ++j) {
      float sum = b2[j];
      for (int i = 0; i < HIDDEN; ++i) sum += w2[j][i] * h1[i];
      h2[j] = tanhf(sum);
    }
    float out = b3;
    for (int i = 0; i < HIDDEN; ++i) out += w3[i] * h2[i];
    if (!isfinite(out)) return NAN;
    return fminf(maxCorrection, fmaxf(minCorrection, out));
  }
};

struct DriveController : Controller {
  DriveCalibration calibration;
  CorrectionModel model;
  StudentModel student;
  DriveMode mode = DriveMode::Pwm;
  uint32_t studentStartedAt = 0, studentAt = 0;
  int64_t studentLeftStart = 0, studentRightStart = 0;
  bool studentStarted = false;
  float speed = 0, correction = 0, measuredLeft = 0, measuredRight = 0;
  float desiredLeft = 0, desiredRight = 0;
  float integralLeft = 0, integralRight = 0;
  uint32_t speedAt = 0;
  int64_t countLeftAt = 0, countRightAt = 0;
  bool speedInitialized = false;

  void resetSpeed() {
    integralLeft = integralRight = 0;
    measuredLeft = measuredRight = desiredLeft = desiredRight = 0;
    speedInitialized = false;
    studentStarted = false;
  }

  void stop(StopReason why) {
    Controller::stop(why);
    resetSpeed();
  }

  bool arm(uint32_t now, int batteryMv) {
    resetSpeed();
    mode = DriveMode::Pwm; speed = correction = 0;
    return Controller::arm(now, batteryMv);
  }

  bool command(uint32_t now, uint32_t commandEpoch, uint32_t seq,
               int newLeft, int newRight, int newServo) {
    if (!Controller::command(now, commandEpoch, seq, newLeft, newRight, newServo)) return false;
    mode = DriveMode::Pwm; speed = correction = 0;
    resetSpeed();
    return true;
  }

  bool drive(uint32_t now, uint32_t commandEpoch, uint32_t seq,
             DriveMode newMode, float newSpeed, float newCorrection, int pwm) {
    if (!armed || commandEpoch != epoch || seq <= sequence) return false;
    if (!isfinite(newSpeed) || !isfinite(newCorrection) || newCorrection < -0.2f ||
        newCorrection > 0.2f || newSpeed < 0 || pwm < 0 || pwm > MAX_PERCENT ||
        (newMode != DriveMode::Pwm && newMode != DriveMode::Speed && newMode != DriveMode::Learned &&
         newMode != DriveMode::Student) ||
        (newMode != DriveMode::Pwm && (!calibration.valid() || newSpeed > calibration.maxSpeed)) ||
        (newMode == DriveMode::Learned && !model.choose(newSpeed, newCorrection)) ||
        (newMode == DriveMode::Student &&
         (!student.valid() || newSpeed < student.minSpeed || newSpeed > student.maxSpeed))) {
      stop(StopReason::InvalidCommand); return false;
    }
    // Only an external command renews the lease. PI updates never call command().
    if (!Controller::command(now, commandEpoch, seq,
                            newMode == DriveMode::Pwm ? pwm : targetLeft,
                            newMode == DriveMode::Pwm ? pwm : targetRight, 0)) return false;
    if (mode != newMode || newSpeed == 0) resetSpeed();
    if (newMode == DriveMode::Student) {
      // Keep the student's own correction; the commanded value is ignored in this mode.
      if (mode != DriveMode::Student) { studentStarted = false; correction = 0; }
      mode = newMode; speed = newSpeed;
    } else {
      mode = newMode; speed = newSpeed; correction = newCorrection;
    }
    if (mode != DriveMode::Pwm && speed == 0) targetLeft = targetRight = 0;
    return true;
  }

  int pi(float target, float measured, float feedforward, float deadband,
         float dt, float& integral) const {
    if (target == 0) { integral = 0; return 0; }
    const float error = target - measured;
    const float candidate = integral + calibration.ki * error * dt;
    const float base = deadband + feedforward * target + calibration.kp * error;
    const float raw = base + candidate;
    if ((raw >= 0 && raw <= MAX_PERCENT) || (raw > MAX_PERCENT && error < 0) ||
        (raw < 0 && error > 0)) integral = candidate;
    const float output = fminf(MAX_PERCENT, fmaxf(0, base + integral));
    return static_cast<int>(lroundf(output));
  }

  void updateStudent(uint32_t now, int64_t leftCount, int64_t rightCount) {
    if (!studentStarted) {
      studentStarted = true;
      studentStartedAt = now; studentAt = now - 100;
      studentLeftStart = leftCount; studentRightStart = rightCount;
    }
    if (static_cast<uint32_t>(now - studentAt) < 100) return;
    studentAt = now;
    const float left = (leftCount - studentLeftStart) * calibration.leftMetersPerCount * calibration.leftSign;
    const float right = (rightCount - studentRightStart) * calibration.rightMetersPerCount * calibration.rightSign;
    const float elapsed = fminf(student.elapsedCap, (now - studentStartedAt) / 1000.0f);
    const float input[StudentModel::INPUTS] = {
      speed, elapsed, measuredLeft, measuredRight, (right - left) / student.trackM};
    const float out = student.predict(input);
    if (!isfinite(out)) { stop(StopReason::InvalidCommand); return; }
    correction = out;
  }

  void tick(uint32_t now, int64_t leftCount, int64_t rightCount, int batteryMv) {
    if (armed && calibration.valid()) {
      if (!speedInitialized) {
        speedAt = now; countLeftAt = leftCount; countRightAt = rightCount;
        speedInitialized = true;
      }
      const uint32_t elapsed = now - speedAt;
      if (elapsed >= 20) {
        const float dt = elapsed / 1000.0f;
        const float rawLeft = (leftCount - countLeftAt) * calibration.leftMetersPerCount * calibration.leftSign / dt;
        const float rawRight = (rightCount - countRightAt) * calibration.rightMetersPerCount * calibration.rightSign / dt;
        const float alpha = dt / (0.06f + dt);
        measuredLeft += alpha * (rawLeft - measuredLeft);
        measuredRight += alpha * (rawRight - measuredRight);
        if (mode == DriveMode::Student) updateStudent(now, leftCount, rightCount);
        if (mode != DriveMode::Pwm && armed) {
          desiredLeft = speed * (1 - correction);
          desiredRight = speed * (1 + correction);
          targetLeft = pi(desiredLeft, measuredLeft, calibration.leftFeedforward,
                          calibration.leftDeadband, dt, integralLeft);
          targetRight = pi(desiredRight, measuredRight, calibration.rightFeedforward,
                           calibration.rightDeadband, dt, integralRight);
        }
        speedAt = now; countLeftAt = leftCount; countRightAt = rightCount;
      }
    }
    Controller::tick(now, leftCount, rightCount, batteryMv);
    if (!armed) resetSpeed();
  }
};
}  // namespace robot
