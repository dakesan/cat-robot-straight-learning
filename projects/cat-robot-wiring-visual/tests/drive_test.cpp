#include <cassert>
#include <cmath>
#include <iostream>
#include "../sketches/cat_robot/drive.h"
#include "../sketches/cat_robot/telemetry.h"

int main() {
  robot::DriveController c;
  assert(c.arm(0, 0));
  assert(!c.drive(1, c.epoch, 1, robot::DriveMode::Speed, 0.1f, 0, 20));
  assert(!c.armed);
  c.calibration.calibrated = true;
  c.calibration.leftMetersPerCount = c.calibration.rightMetersPerCount = 0.0001f;
  c.calibration.leftFeedforward = c.calibration.rightFeedforward = 200;
  c.calibration.maxSpeed = 0.3f;
  assert(c.calibration.valid());
  assert(c.arm(0, 0));
  assert(c.drive(1, c.epoch, 1, robot::DriveMode::Speed, 0.1f, 0, 20));
  for (unsigned t = 10; t < 300; t += 10) {
    c.tick(t, t, t, 0);
    assert(c.armed && c.left <= 50 && c.right <= 50 && c.servo == 0);
  }
  assert(c.lastCommand == 1 && c.sequence == 1);
  c.tick(301, 301, 301, 0);
  assert(!c.armed && c.left == 0 && c.reason == robot::StopReason::CommandTimeout);
  c.model.fitted = true;
  c.model.minSpeed = 0.05f; c.model.maxSpeed = 0.3f;
  c.model.intercept[0] = 0.05f;
  c.model.coefficients[0][1] = 1;
  float delta = 0;
  assert(c.model.choose(0.1f, delta) && std::abs(delta + 0.05f) < 1e-5);
  assert(!c.model.choose(0.4f, delta));
  assert(c.arm(400, 0));
  assert(c.drive(401, c.epoch, 1, robot::DriveMode::Learned, 0.1f, 0, 20));
  for (unsigned t = 410; t <= 1600 && c.armed; t += 10) {
    if (t % 100 == 0) assert(c.drive(t, c.epoch, c.sequence + 1, robot::DriveMode::Learned, 0.1f, 0, 20));
    c.tick(t, 0, 0, 0);
  }
  assert(!c.armed && c.reason == robot::StopReason::EncoderLeft && c.left == 0);
  robot::Telemetry<4> ring;
  for (int i = 0; i < 7; ++i) { robot::Sample s; s.timeMs = i * 20; ring.push(s); }
  robot::Sample out[4]; uint32_t lost;
  assert(ring.read(0, out, 4, lost) == 4 && lost == 3 && out[0].sample == 4);
  assert(ring.read(7, out, 4, lost) == 0 && lost == 0);
  assert(robot::syncLight(0) && !robot::syncLight(200) && robot::syncLight(600));
  // Student corrector: rejected until fitted, clamped to its range, updated every 100 ms,
  // and never keeps an expired command alive.
  robot::DriveController d;
  d.calibration = c.calibration;
  assert(d.arm(0, 0));
  assert(!d.drive(1, d.epoch, 1, robot::DriveMode::Student, 0.1f, 0, 20));
  assert(!d.armed);
  d.student.fitted = true; d.student.measured = true;
  d.student.minSpeed = 0.05f; d.student.maxSpeed = 0.25f;
  d.student.minCorrection = -0.03f; d.student.maxCorrection = 0.03f;
  d.student.b3 = 0.5f;  // raw output far above the range: must clamp to 0.03
  assert(d.student.valid());
  assert(d.arm(1000, 0));
  assert(!d.drive(1001, d.epoch, 1, robot::DriveMode::Student, 0.3f, 0, 20));
  assert(d.arm(1100, 0));
  assert(d.drive(1101, d.epoch, 1, robot::DriveMode::Student, 0.1f, 0.15f, 20));
  for (unsigned t = 1110; t < 1600; t += 10) {
    if (t % 100 == 0 && t < 1300) assert(d.drive(t, d.epoch, d.sequence + 1, robot::DriveMode::Student, 0.1f, 0, 20));
    d.tick(t, (t - 1100) * 2, -(static_cast<int64_t>(t) - 1100) * 2, 0);
    if (d.armed) assert(d.correction <= 0.03f + 1e-6f && d.correction >= -0.03f - 1e-6f && d.left <= 50 && d.right <= 50);
  }
  // Last external command at t=1200; internal student updates must not extend it.
  assert(!d.armed && d.reason == robot::StopReason::CommandTimeout);
  assert(d.arm(2000, 0));
  assert(d.drive(2001, d.epoch, 1, robot::DriveMode::Student, 0.1f, 0, 20));
  d.tick(2021, 40, -40, 0);  // first tick only starts the speed estimate
  d.tick(2041, 80, -80, 0);
  assert(d.armed && std::abs(d.correction - 0.03f) < 1e-6f);
  std::cout << "PASS: calibration, PI lease, bounds, learned choice, encoder stop, telemetry loss, sync, student\n";
}
