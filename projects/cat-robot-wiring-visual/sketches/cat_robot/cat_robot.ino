#include <Arduino.h>
#include <Preferences.h>
#include <WebServer.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <esp_random.h>
#include <errno.h>
#include "control.h"
#include "control_page.h"
#include "drive.h"
#include "telemetry.h"
#include "learned_model.h"
#include "student_model.h"

constexpr char VERSION[] = "cat-robot/1.4.0";
constexpr uint8_t LEFT_PHASE = 4, LEFT_ENABLE = 5;
constexpr uint8_t RIGHT_PHASE = 6, RIGHT_ENABLE = 7;
constexpr uint8_t LEFT_A = 15, LEFT_B = 16, RIGHT_A = 17, RIGHT_B = 18;
constexpr uint8_t SERVO_PIN = 13, BATTERY_PIN = 1;
constexpr uint8_t SYNC_LED_PIN = 8;
// Onboard addressable RGB LED of generic ESP32-S3 boards: GPIO48 (DevKitC-1 v1.0) or
// GPIO38 (v1.1). Both are unused here, so the sync pattern is written to both in red.
constexpr uint8_t SYNC_RGB_PINS[] = {48, 38};
constexpr uint32_t MOTOR_HZ = 20000, SERVO_HZ = 50;
constexpr uint8_t MOTOR_BITS = 8, SERVO_BITS = 14;

Preferences preferences;
WebServer server(80);
SemaphoreHandle_t controlMutex;
portMUX_TYPE encoderMux = portMUX_INITIALIZER_UNLOCKED;
robot::DriveController controller;
robot::Telemetry<> telemetry;
bool syncEnabled = false, syncActive = false, rgbLight = false;
uint32_t syncStartedAt = 0, lastSampleAt = 0, bootId = 0;
volatile int64_t leftCount = 0, rightCount = 0;
volatile uint32_t leftInvalid = 0, rightInvalid = 0;
volatile uint8_t leftPrevious = 0, rightPrevious = 0;
int batteryMv = 0;
bool outputsReady = false, networkReady = false;
bool leftAttached = false, rightAttached = false, servoAttached = false;
String ssid, wifiPassword, controlToken;
// Optional home network (station mode). The robot's own access point stays available.
String stationSsid, stationPassword;

void ARDUINO_ISR_ATTR leftEncoder() {
  portENTER_CRITICAL_ISR(&encoderMux);
  const uint8_t current = (digitalRead(LEFT_A) << 1) | digitalRead(LEFT_B);
  if ((leftPrevious ^ current) == 3) ++leftInvalid;
  leftCount += robot::encoderDelta(leftPrevious, current);
  leftPrevious = current;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void ARDUINO_ISR_ATTR rightEncoder() {
  portENTER_CRITICAL_ISR(&encoderMux);
  const uint8_t current = (digitalRead(RIGHT_A) << 1) | digitalRead(RIGHT_B);
  if ((rightPrevious ^ current) == 3) ++rightInvalid;
  rightCount += robot::encoderDelta(rightPrevious, current);
  rightPrevious = current;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void forceOutputsLow() {
  if (leftAttached) { ledcWrite(LEFT_ENABLE, 0); ledcDetach(LEFT_ENABLE); leftAttached = false; }
  if (rightAttached) { ledcWrite(RIGHT_ENABLE, 0); ledcDetach(RIGHT_ENABLE); rightAttached = false; }
  if (servoAttached) { ledcWrite(SERVO_PIN, 0); ledcDetach(SERVO_PIN); servoAttached = false; }
  for (uint8_t pin : {LEFT_ENABLE, RIGHT_ENABLE, SERVO_PIN}) {
    digitalWrite(pin, LOW);
    pinMode(pin, OUTPUT);
  }
}

bool applyOutputs(const robot::Controller& state) {
  const bool leftReverse = (state.left < 0) != state.invertLeft;
  const bool rightReverse = (state.right < 0) != state.invertRight;
  // Always remove drive before changing the direction output.
  if ((digitalRead(LEFT_PHASE) != leftReverse) && !ledcWrite(LEFT_ENABLE, 0)) return false;
  if ((digitalRead(RIGHT_PHASE) != rightReverse) && !ledcWrite(RIGHT_ENABLE, 0)) return false;
  digitalWrite(LEFT_PHASE, leftReverse);
  digitalWrite(RIGHT_PHASE, rightReverse);
  const uint32_t leftDuty = robot::motorDuty(state.left);
  const uint32_t rightDuty = robot::motorDuty(state.right);
  const uint32_t servoDuty = state.servo ? (state.servo * 16383UL + 10000) / 20000 : 0;
  return ledcWrite(LEFT_ENABLE, leftDuty) && ledcWrite(RIGHT_ENABLE, rightDuty) &&
         ledcWrite(SERVO_PIN, servoDuty);
}

void controlTask(void*) {
  TickType_t wake = xTaskGetTickCount();
  uint32_t batteryReadAt = 0;
  for (;;) {
    int64_t leftNow, rightNow;
    uint32_t leftBad, rightBad;
    portENTER_CRITICAL(&encoderMux);
    leftNow = leftCount; rightNow = rightCount;
    leftBad = leftInvalid; rightBad = rightInvalid;
    portEXIT_CRITICAL(&encoderMux);
    const uint32_t now = millis();
    bool readBattery;
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    readBattery = controller.batteryEnabled;
    xSemaphoreGive(controlMutex);
    int sampledMv = -1;
    if (readBattery && static_cast<uint32_t>(now - batteryReadAt) >= 50) {
      sampledMv = analogReadMilliVolts(BATTERY_PIN) * 4;
      batteryReadAt = now;
    }
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    if (sampledMv >= 0) batteryMv = sampledMv;
    controller.tick(now, leftNow, rightNow, batteryMv);
    if (outputsReady && !applyOutputs(controller)) {
      outputsReady = false;
      controller.stop(robot::StopReason::OutputError);
      forceOutputsLow();
    }
    bool light = syncEnabled && syncActive && robot::syncLight(now - syncStartedAt);
    if (syncEnabled) digitalWrite(SYNC_LED_PIN, light ? HIGH : LOW);
    if (light != rgbLight) {
      // Only on changes: each write is a short RMT transfer.
      for (uint8_t pin : SYNC_RGB_PINS) rgbLedWrite(pin, light ? 255 : 0, 0, 0);
      rgbLight = light;
    }
    if (syncActive && static_cast<uint32_t>(now - syncStartedAt) >= 2000) syncActive = false;
    if (static_cast<uint32_t>(now - lastSampleAt) >= 20) {
      robot::Sample sample;
      sample.timeMs = now; sample.epoch = controller.epoch;
      sample.commandSequence = controller.sequence;
      sample.encoderLeft = leftNow; sample.encoderRight = rightNow;
      sample.invalidLeft = leftBad; sample.invalidRight = rightBad;
      sample.left = controller.left; sample.right = controller.right;
      sample.targetLeft = controller.targetLeft; sample.targetRight = controller.targetRight;
      sample.speed = controller.speed; sample.correction = controller.correction;
      sample.measuredLeft = controller.measuredLeft; sample.measuredRight = controller.measuredRight;
      sample.mode = controller.mode; sample.armed = controller.armed;
      sample.reason = controller.reason; sample.syncLed = light;
      sample.batteryMv = controller.batteryEnabled ? batteryMv : -1;
      telemetry.push(sample); lastSampleAt = now;
    }
    xSemaphoreGive(controlMutex);
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(10));
  }
}

String statusJson() {
  robot::DriveController state;
  int voltage;
  bool outputOK;
  uint32_t dutyLeft, dutyRight, dutyServo;
  xSemaphoreTake(controlMutex, portMAX_DELAY);
  state = controller; voltage = batteryMv; outputOK = outputsReady;
  dutyLeft = outputsReady ? ledcRead(LEFT_ENABLE) : 0;
  dutyRight = outputsReady ? ledcRead(RIGHT_ENABLE) : 0;
  dutyServo = outputsReady ? ledcRead(SERVO_PIN) : 0;
  xSemaphoreGive(controlMutex);
  int64_t leftNow, rightNow;
  uint32_t leftBad, rightBad;
  portENTER_CRITICAL(&encoderMux);
  leftNow = leftCount; rightNow = rightCount;
  leftBad = leftInvalid; rightBad = rightInvalid;
  portEXIT_CRITICAL(&encoderMux);
  String json;
  json.reserve(650);
  json = "{\"firmware\":\"" + String(VERSION) + "\",\"armed\":" + (state.armed ? "true" : "false");
  json += ",\"outputs_ready\":" + String(outputOK ? "true" : "false");
  json += ",\"reason\":\"" + String(robot::reasonName(state.reason)) + "\"";
  json += ",\"epoch\":" + String(state.epoch) + ",\"sequence\":" + String(state.sequence);
  json += ",\"left\":" + String(state.left) + ",\"right\":" + String(state.right);
  json += ",\"servo_us\":" + String(state.servo);
  json += ",\"pwm_left\":" + String(dutyLeft) + ",\"pwm_right\":" + String(dutyRight);
  json += ",\"pwm_servo\":" + String(dutyServo);
  json += ",\"encoder_left\":" + String(static_cast<long long>(leftNow));
  json += ",\"encoder_right\":" + String(static_cast<long long>(rightNow));
  json += ",\"encoder_left_invalid\":" + String(leftBad) + ",\"encoder_right_invalid\":" + String(rightBad);
  json += ",\"invert_left\":" + String(state.invertLeft ? "true" : "false");
  json += ",\"invert_right\":" + String(state.invertRight ? "true" : "false");
  json += ",\"battery_enabled\":" + String(state.batteryEnabled ? "true" : "false");
  json += ",\"battery_mv\":" + (state.batteryEnabled ? String(voltage) : String("null"));
  json += ",\"boot_id\":" + String(bootId) + ",\"time_ms\":" + String(millis());
  json += ",\"calibrated\":" + String(state.calibration.valid() ? "true" : "false");
  json += ",\"model_id\":\"" + String(deployedModelId()) + "\"";
  json += ",\"model_measured\":" + String(state.model.measured ? "true" : "false");
  json += ",\"student_id\":\"" + String(deployedStudentId()) + "\"";
  json += ",\"student_measured\":" + String(state.student.measured ? "true" : "false");
  json += ",\"station_ssid_set\":" + String(stationSsid.length() ? "true" : "false");
  json += ",\"station_connected\":" + String(WiFi.status() == WL_CONNECTED ? "true" : "false");
  json += ",\"station_ip\":\"" + (WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : String()) + "\"";
  json += ",\"command_timeout_ms\":300,\"encoder_timeout_ms\":800,\"max_percent\":50}";
  return json;
}

void reply(int code, const String& body) {
  server.sendHeader("Cache-Control", "no-store");
  server.send(code, "application/json", body);
}

bool authorized() {
  if (server.header("X-Robot-Token") == controlToken) return true;
  reply(403, "{\"error\":\"control token required\"}");
  return false;
}

bool numberArgument(const char* name, long minimum, long maximum, long& result) {
  if (!server.hasArg(name)) return false;
  const String value = server.arg(name);
  if (value.isEmpty() || value.length() > 10) return false;
  for (size_t i = 0; i < value.length(); ++i)
    if ((value[i] < '0' || value[i] > '9') && !(i == 0 && value[i] == '-')) return false;
  char* end;
  errno = 0;
  const long parsed = strtol(value.c_str(), &end, 10);
  if (errno || *end || parsed < minimum || parsed > maximum) return false;
  result = parsed;
  return true;
}

bool floatArgument(const char* name, float& result) {
  if (!server.hasArg(name)) return false;
  const String value = server.arg(name);
  if (value.isEmpty() || value.length() > 32) return false;
  char* end;
  errno = 0;
  result = strtof(value.c_str(), &end);
  return !errno && end != value.c_str() && !*end && isfinite(result);
}

String sampleJson(const robot::Sample& s) {
  String json; json.reserve(600);
  json = "{\"sample\":" + String(s.sample) + ",\"time_ms\":" + String(s.timeMs);
  json += ",\"epoch\":" + String(s.epoch) + ",\"sequence\":" + String(s.commandSequence);
  json += ",\"encoder_left\":" + String(static_cast<long long>(s.encoderLeft));
  json += ",\"encoder_right\":" + String(static_cast<long long>(s.encoderRight));
  json += ",\"encoder_left_invalid\":" + String(s.invalidLeft) + ",\"encoder_right_invalid\":" + String(s.invalidRight);
  json += ",\"left\":" + String(s.left) + ",\"right\":" + String(s.right);
  json += ",\"target_left\":" + String(s.targetLeft) + ",\"target_right\":" + String(s.targetRight);
  json += ",\"speed_mps\":" + String(s.speed, 6) + ",\"correction\":" + String(s.correction, 6);
  json += ",\"measured_left_mps\":" + String(s.measuredLeft, 6);
  json += ",\"measured_right_mps\":" + String(s.measuredRight, 6);
  json += ",\"mode\":" + String(static_cast<int>(s.mode));
  json += ",\"armed\":" + String(s.armed ? "true" : "false");
  json += ",\"reason\":\"" + String(robot::reasonName(s.reason)) + "\"";
  json += ",\"sync_led\":" + String(s.syncLed ? "true" : "false");
  json += ",\"battery_mv\":" + (s.batteryMv < 0 ? String("null") : String(s.batteryMv)) + "}";
  return json;
}

String telemetryJson(uint32_t after, bool tail = false) {
  robot::Sample samples[16]; uint32_t dropped, latest;
  xSemaphoreTake(controlMutex, portMAX_DELAY);
  latest = telemetry.sequence;
  if (tail) after = latest > 16 ? latest - 16 : 0;
  const unsigned count = telemetry.read(after, samples, 16, dropped);
  xSemaphoreGive(controlMutex);
  String json = "{\"boot_id\":" + String(bootId) + ",\"latest\":" + String(latest);
  json += ",\"dropped\":" + String(dropped) + ",\"samples\":[";
  for (unsigned i = 0; i < count; ++i) {
    if (i) json += ',';
    json += sampleJson(samples[i]);
  }
  return json + "]}";
}

void setupExperimentWeb() {
  server.on("/api/telemetry", HTTP_GET, [] {
    long after = 0;
    if (server.hasArg("after") && !numberArgument("after", 0, 2147483647, after)) {
      reply(400, "{\"error\":\"invalid cursor\"}"); return;
    }
    reply(200, telemetryJson(after));
  });
  server.on("/api/drive", HTTP_POST, [] {
    if (!authorized()) return;
    long epoch, sequence, mode, pwm;
    float speed, correction;
    if (!numberArgument("epoch", 1, 2147483647, epoch) ||
        !numberArgument("seq", 1, 2147483647, sequence) || !numberArgument("mode", 0, 3, mode) ||
        !numberArgument("pwm", 0, 50, pwm) || !floatArgument("speed_mps", speed) ||
        !floatArgument("correction", correction)) {
      xSemaphoreTake(controlMutex, portMAX_DELAY);
      controller.stop(robot::StopReason::InvalidCommand);
      xSemaphoreGive(controlMutex);
      reply(400, "{\"error\":\"invalid drive command\"}"); return;
    }
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    const bool accepted = controller.drive(millis(), epoch, sequence,
      static_cast<robot::DriveMode>(mode), speed, correction, pwm);
    xSemaphoreGive(controlMutex);
    reply(accepted ? 200 : 409, accepted ? "{\"ok\":true}" : "{\"error\":\"stopped, expired, uncalibrated or model unavailable\"}");
  });
  server.on("/api/calibration", HTTP_POST, [] {
    if (!authorized()) return;
    robot::DriveCalibration cal; long confirmed, signLeft, signRight;
    const bool valid = numberArgument("confirmed", 1, 1, confirmed) &&
      numberArgument("left_sign", -1, 1, signLeft) && numberArgument("right_sign", -1, 1, signRight) &&
      floatArgument("left_meters_per_count", cal.leftMetersPerCount) &&
      floatArgument("right_meters_per_count", cal.rightMetersPerCount) &&
      floatArgument("left_feedforward", cal.leftFeedforward) && floatArgument("right_feedforward", cal.rightFeedforward) &&
      floatArgument("left_deadband", cal.leftDeadband) && floatArgument("right_deadband", cal.rightDeadband) &&
      floatArgument("kp", cal.kp) && floatArgument("ki", cal.ki) && floatArgument("max_speed", cal.maxSpeed);
    cal.calibrated = true; cal.leftSign = valid ? signLeft : 0; cal.rightSign = valid ? signRight : 0;
    if (!valid || !cal.valid()) { reply(400, "{\"error\":\"invalid measured calibration\"}"); return; }
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    if (controller.armed) { xSemaphoreGive(controlMutex); reply(409, "{\"error\":\"stop before calibration\"}"); return; }
    const bool saved = preferences.putBytes("drivecal", &cal, sizeof(cal)) == sizeof(cal);
    if (saved) controller.calibration = cal;
    xSemaphoreGive(controlMutex);
    reply(saved ? 200 : 500, saved ? "{\"ok\":true}" : "{\"error\":\"persistence failed\"}");
  });
  server.on("/api/sync", HTTP_POST, [] {
    if (!authorized()) return;
    long enabled;
    if (!numberArgument("enabled", 0, 1, enabled)) { reply(400, "{\"error\":\"confirm LED on GPIO8\"}"); return; }
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    if (controller.armed) { xSemaphoreGive(controlMutex); reply(409, "{\"error\":\"stop before sync\"}"); return; }
    syncEnabled = enabled; syncActive = enabled;
    digitalWrite(SYNC_LED_PIN, LOW);
    if (enabled) { pinMode(SYNC_LED_PIN, OUTPUT); syncStartedAt = millis(); }
    else pinMode(SYNC_LED_PIN, INPUT);
    xSemaphoreGive(controlMutex);
    reply(200, "{\"ok\":true}");
  });
}

void setupWeb() {
  const char* headers[] = {"X-Robot-Token"};
  server.collectHeaders(headers, 1);
  server.on("/", HTTP_GET, [] {
    String page = FPSTR(CONTROL_PAGE);
    page.replace("__TOKEN__", controlToken);
    server.sendHeader("Cache-Control", "no-store");
    server.sendHeader("X-Frame-Options", "DENY");
    server.send(200, "text/html; charset=utf-8", page);
  });
  server.on("/api/status", HTTP_GET, [] { reply(200, statusJson()); });
  setupExperimentWeb();
  server.on("/api/stop", HTTP_POST, [] {
    if (!authorized()) return;
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    controller.stop(robot::StopReason::Operator);
    xSemaphoreGive(controlMutex);
    reply(200, statusJson());
  });
  server.on("/api/arm", HTTP_POST, [] {
    if (!authorized()) return;
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    const bool accepted = outputsReady && networkReady && !controller.armed &&
                          controller.arm(millis(), batteryMv);
    xSemaphoreGive(controlMutex);
    if (!accepted) { reply(409, "{\"error\":\"cannot arm; check stop reason or stop first\"}"); return; }
    reply(200, statusJson());
  });
  server.on("/api/command", HTTP_POST, [] {
    if (!authorized()) return;
    long epoch, seq, left, right, servo;
    if (!numberArgument("epoch", 1, 2147483647, epoch) || !numberArgument("seq", 1, 2147483647, seq) ||
        !numberArgument("left", -50, 50, left) || !numberArgument("right", -50, 50, right) ||
        !numberArgument("servo", 0, 1600, servo) || (servo != 0 && servo < 1400)) {
      xSemaphoreTake(controlMutex, portMAX_DELAY);
      controller.stop(robot::StopReason::InvalidCommand);
      xSemaphoreGive(controlMutex);
      reply(400, "{\"error\":\"invalid command\"}"); return;
    }
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    const bool accepted = controller.command(millis(), epoch, seq, left, right, servo);
    xSemaphoreGive(controlMutex);
    reply(accepted ? 200 : 409, accepted ? "{\"ok\":true}" : "{\"error\":\"stopped, expired or old command\"}");
  });
  server.on("/api/config", HTTP_POST, [] {
    if (!authorized()) return;
    long left, right, battery;
    if (!numberArgument("invert_left", 0, 1, left) || !numberArgument("invert_right", 0, 1, right) ||
        !numberArgument("battery_enabled", 0, 1, battery)) {
      reply(400, "{\"error\":\"invalid configuration\"}"); return;
    }
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    if (controller.armed) { xSemaphoreGive(controlMutex); reply(409, "{\"error\":\"stop before saving\"}"); return; }
    controller.invertLeft = left;
    controller.invertRight = right;
    controller.batteryEnabled = battery;
    batteryMv = 0;
    xSemaphoreGive(controlMutex);
    const bool saved = preferences.putBool("invL", left) == 1 &&
                       preferences.putBool("invR", right) == 1 && preferences.putBool("battery", battery) == 1;
    if (!saved) { reply(500, "{\"error\":\"configuration persistence failed\"}"); return; }
    reply(200, statusJson());
  });
  server.onNotFound([] { reply(404, "{\"error\":\"not found\"}"); });
  server.begin();
}

bool startSync() {
  xSemaphoreTake(controlMutex, portMAX_DELAY);
  const bool allowed = !controller.armed;
  if (allowed) {
    syncEnabled = syncActive = true;
    digitalWrite(SYNC_LED_PIN, LOW);
    pinMode(SYNC_LED_PIN, OUTPUT);
    syncStartedAt = millis();
  }
  xSemaphoreGive(controlMutex);
  return allowed;
}

void serialCommand(const char* line) {
  if (strcmp(line, "status") == 0 || strcmp(line, "?") == 0) Serial.println(statusJson());
  else if (strcmp(line, "sync") == 0)
    Serial.println(startSync() ? "{\"sync\":\"started\"}" : "ERROR: stop before sync");
  else if (strcmp(line, "telemetry") == 0) Serial.println(telemetryJson(0, true));
  else if (strcmp(line, "wifi") == 0) {
    Serial.println("SSID=" + ssid);
    Serial.println("PASSWORD=" + wifiPassword);
    Serial.println("URL=http://192.168.4.1/");
    Serial.println("STATION_SSID=" + stationSsid);
    Serial.println("STATION_IP=" + (WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : String()));
  } else if (strncmp(line, "sta-ssid ", 9) == 0 || strncmp(line, "sta-pass ", 9) == 0 ||
             strcmp(line, "sta-clear") == 0) {
    // Home network credentials are stored only while stopped and take effect after reboot.
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    const bool armed = controller.armed;
    xSemaphoreGive(controlMutex);
    if (armed) { Serial.println("ERROR: stop before changing the network"); return; }
    bool saved;
    if (strcmp(line, "sta-clear") == 0) {
      preferences.remove("sta-ssid");
      preferences.remove("sta-pass");
      saved = !preferences.isKey("sta-ssid") && !preferences.isKey("sta-pass");
    }
    else if (line[4] == 's')
      saved = preferences.putString("sta-ssid", line + 9) > 0;
    else
      saved = preferences.putString("sta-pass", line + 9) > 0;
    Serial.println(saved ? "{\"network\":\"saved; reboot to apply\"}" : "ERROR: network persistence failed");
  } else if (strcmp(line, "stop") == 0) {
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    controller.stop(robot::StopReason::Operator);
    xSemaphoreGive(controlMutex);
    Serial.println(statusJson());
  } else if (strlen(line) == 10 && strncmp(line, "invert ", 7) == 0 && line[8] == ' ' &&
             (line[7] == '0' || line[7] == '1') && (line[9] == '0' || line[9] == '1')) {
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    if (controller.armed) {
      xSemaphoreGive(controlMutex);
      Serial.println("ERROR: stop before changing direction");
      return;
    }
    const bool left = line[7] == '1', right = line[9] == '1';
    const bool saved = preferences.putBool("invL", left) == 1 && preferences.putBool("invR", right) == 1;
    if (saved) { controller.invertLeft = left; controller.invertRight = right; }
    xSemaphoreGive(controlMutex);
    if (saved) Serial.println(statusJson());
    else Serial.println("ERROR: direction persistence failed");
  } else Serial.println("Commands: status, telemetry, sync, wifi, sta-ssid <name>, sta-pass <password>, sta-clear, stop, invert <left:0|1> <right:0|1>, help");
}

void setup() {
  forceOutputsLow();
  for (uint8_t pin : SYNC_RGB_PINS) rgbLedWrite(pin, 0, 0, 0);
  for (uint8_t pin : {LEFT_PHASE, RIGHT_PHASE}) {
    digitalWrite(pin, LOW); pinMode(pin, OUTPUT);
  }
  Serial.begin(115200);
  controlMutex = xSemaphoreCreateMutex();
  if (!controlMutex) { forceOutputsLow(); for (;;) delay(1000); }
  const bool preferencesOK = preferences.begin("cat-robot", false);
  if (preferencesOK) {
    controller.invertLeft = preferences.getBool("invL", false);
    controller.invertRight = preferences.getBool("invR", false);
    controller.batteryEnabled = preferences.getBool("battery", false);
    if (preferences.getBytesLength("drivecal") == sizeof(robot::DriveCalibration)) {
      robot::DriveCalibration cal;
      preferences.getBytes("drivecal", &cal, sizeof(cal));
      if (cal.valid()) controller.calibration = cal;
    }
  }
  controller.model = deployedModel();
  if (!controller.model.measured) controller.model.fitted = false;
  controller.student = deployedStudent();
  if (!controller.student.measured) controller.student.fitted = false;
  bootId = esp_random();
  for (uint8_t pin : {LEFT_A, LEFT_B, RIGHT_A, RIGHT_B}) pinMode(pin, INPUT_PULLUP);
  leftPrevious = (digitalRead(LEFT_A) << 1) | digitalRead(LEFT_B);
  rightPrevious = (digitalRead(RIGHT_A) << 1) | digitalRead(RIGHT_B);
  attachInterrupt(LEFT_A, leftEncoder, CHANGE); attachInterrupt(LEFT_B, leftEncoder, CHANGE);
  attachInterrupt(RIGHT_A, rightEncoder, CHANGE); attachInterrupt(RIGHT_B, rightEncoder, CHANGE);
  analogReadResolution(12);
  analogSetPinAttenuation(BATTERY_PIN, ADC_11db);
  leftAttached = ledcAttachChannel(LEFT_ENABLE, MOTOR_HZ, MOTOR_BITS, 0);
  rightAttached = ledcAttachChannel(RIGHT_ENABLE, MOTOR_HZ, MOTOR_BITS, 1);
  servoAttached = ledcAttachChannel(SERVO_PIN, SERVO_HZ, SERVO_BITS, 2);
  outputsReady = leftAttached && rightAttached && servoAttached;
  if (outputsReady) outputsReady = applyOutputs(controller);
  if (!outputsReady) { forceOutputsLow(); controller.stop(robot::StopReason::OutputError); }
  if (xTaskCreatePinnedToCore(controlTask, "robot-control", 4096, nullptr, 3, nullptr, 1) != pdPASS) {
    outputsReady = false; forceOutputsLow(); controller.stop(robot::StopReason::OutputError);
  }

  const uint64_t chipId = ESP.getEfuseMac();
  char networkName[32];
  snprintf(networkName, sizeof(networkName), "CatRobot-%06lX", static_cast<unsigned long>((chipId >> 24) & 0xFFFFFF));
  ssid = networkName;
  stationSsid = preferencesOK ? preferences.getString("sta-ssid", "") : String();
  stationPassword = preferencesOK ? preferences.getString("sta-pass", "") : String();
  WiFi.mode(stationSsid.length() ? WIFI_AP_STA : WIFI_AP);
  wifiPassword = preferencesOK ? preferences.getString("ap-pass", "") : String();
  if (wifiPassword.length() != 16) {
    static constexpr char alphabet[] = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
    wifiPassword = "";
    for (int i = 0; i < 16; ++i) wifiPassword += alphabet[esp_random() % (sizeof(alphabet) - 1)];
    if (preferencesOK) preferences.putString("ap-pass", wifiPassword);
  }
  for (int i = 0; i < 4; ++i) {
    char value[9]; snprintf(value, sizeof(value), "%08lx", static_cast<unsigned long>(esp_random()));
    controlToken += value;
  }
  WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), IPAddress(255, 255, 255, 0));
  networkReady = WiFi.softAP(ssid.c_str(), wifiPassword.c_str(), 6, false, 1);
  if (networkReady && stationSsid.length()) {
    WiFi.setHostname("catrobot");
    WiFi.setAutoReconnect(true);
    WiFi.begin(stationSsid.c_str(), stationPassword.c_str());
    // Station modem sleep delays replies by up to ~1 s, beyond the 300 ms command lease.
    WiFi.setSleep(false);
    MDNS.begin("catrobot");
  }
  if (networkReady) setupWeb();
  else {
    xSemaphoreTake(controlMutex, portMAX_DELAY);
    controller.stop(robot::StopReason::NetworkError);
    xSemaphoreGive(controlMutex);
  }
  Serial.println(VERSION);
}

void loop() {
  if (networkReady) server.handleClient();
  static char line[128];
  static size_t length = 0;
  static bool overflow = false;
  for (int i = 0; i < 64 && Serial.available(); ++i) {
    const char value = Serial.read();
    if (value == '\r' || value == '\n') {
      if (!overflow && length) { line[length] = 0; serialCommand(line); }
      length = 0; overflow = false;
    } else if (length + 1 < sizeof(line)) line[length++] = value;
    else overflow = true;
  }
  delay(1);
}
