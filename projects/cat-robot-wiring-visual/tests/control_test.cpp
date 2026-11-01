#include <cassert>
#include <cstdint>
#include <climits>
#include <iostream>
#include "../sketches/cat_robot/control.h"

int main() {
  using robot::Controller;
  using robot::StopReason;
  Controller c;
  assert(!c.armed && c.left == 0 && c.right == 0 && c.servo == 0);
  assert(!c.command(0, 1, 1, 20, 20, 1500));
  assert(c.arm(100, 0));
  const auto oldEpoch = c.epoch;
  assert(c.command(110, c.epoch, 1, 20, -20, 1600));
  c.tick(120, 0, 0, 0);
  assert(c.left == 2 && c.right == -2 && c.servo == 1510);
  assert(!c.command(130, c.epoch, 1, -50, 50, 1400));
  assert(c.targetLeft == 20 && c.targetRight == -20);
  c.tick(410, 0, 0, 0);
  assert(!c.armed && c.left == 0 && c.right == 0 && c.servo == 0);
  assert(c.reason == StopReason::CommandTimeout);
  assert(!c.command(420, c.epoch, 2, 50, 50, 1600));
  assert(c.arm(500, 0));
  assert(!c.command(510, oldEpoch, 100, 50, 50, 1600));
  assert(c.command(510, c.epoch, 1, 20, 0, 0));
  for (uint32_t t = 520; t <= 1400; t += 10) {
    if (t % 100 == 0 && c.armed) c.command(t, c.epoch, c.sequence + 1, 20, 0, 0);
    c.tick(t, 0, 0, 0);
  }
  assert(!c.armed && c.reason == StopReason::EncoderLeft && c.left == 0);

  assert(c.arm(1500, 0));
  assert(c.command(1510, c.epoch, 1, 20, 20, 1400));
  for (uint32_t t = 1520; t <= 2800; t += 10) {
    if (t % 100 == 0) assert(c.command(t, c.epoch, c.sequence + 1, 20, 20, 1400));
    c.tick(t, t, t, 0);
  }
  assert(c.armed && c.left == 20 && c.right == 20 && c.servo == 1400);
  assert(c.command(2810, c.epoch, c.sequence + 1, -20, 0, 0));
  c.tick(2820, 2820, 2820, 0);
  assert(c.left == 18 && c.right == 0 && c.servo == 0);
  for (uint32_t t = 2830; t <= 2910; t += 10) c.tick(t, t, t, 0);
  assert(c.left == 0);
  c.tick(2920, 2920, 2920, 0);
  assert(c.left == -2);
  assert(!c.command(2930, c.epoch, c.sequence + 1, INT_MIN, 0, 0));
  assert(!c.armed && c.reason == StopReason::InvalidCommand);

  c.batteryEnabled = true;
  assert(!c.arm(3000, 0) && c.reason == StopReason::BatterySensor);
  assert(!c.arm(3000, 6700) && c.reason == StopReason::BatteryLow);
  assert(c.arm(3000, 8000));
  c.tick(3010, 0, 0, 6700);
  c.tick(3209, 0, 0, 6700);
  assert(c.armed);
  c.tick(3210, 0, 0, 6700);
  assert(!c.armed && c.reason == StopReason::BatteryLow);
  c.batteryEnabled = false;
  assert(c.arm(UINT32_MAX - 100, 0));
  c.tick(198, 0, 0, 0);
  assert(c.armed);
  c.tick(199, 0, 0, 0);
  assert(!c.armed && c.reason == StopReason::CommandTimeout);
  assert(c.arm(4000, 0));
  assert(!c.command(4300, c.epoch, 1, 10, 10, 0));
  assert(!c.armed);
  int total = 0;
  for (const auto edge : {1, 3, 2, 0}) {
    static uint8_t previous = 0;
    total += robot::encoderDelta(previous, edge);
    previous = edge;
  }
  assert(total == -4 && robot::encoderDelta(0, 3) == 0);
  const int8_t expected[] = {0,-1,1,0,1,0,0,-1,-1,0,0,1,0,1,-1,0};
  for (uint8_t previous = 0; previous < 4; ++previous)
    for (uint8_t current = 0; current < 4; ++current)
      assert(robot::encoderDelta(previous, current) == expected[(previous << 2) | current]);
  std::cout << "PASS: startup, command lease, replay, epoch, encoder timeout, ramp, servo, battery, wraparound\n";
}
