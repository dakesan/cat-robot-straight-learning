#include <array>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "drive.h"

namespace py = pybind11;
PYBIND11_MODULE(_native, m) {
  m.attr("source_sha") = CONTROL_SOURCE_SHA;
  py::enum_<robot::DriveMode>(m, "Mode")
    .value("pwm", robot::DriveMode::Pwm).value("speed", robot::DriveMode::Speed)
    .value("learned", robot::DriveMode::Learned).value("student", robot::DriveMode::Student);
  py::class_<robot::DriveCalibration>(m, "Calibration")
    .def(py::init<>())
    .def_readwrite("calibrated", &robot::DriveCalibration::calibrated)
    .def_readwrite("left_meters_per_count", &robot::DriveCalibration::leftMetersPerCount)
    .def_readwrite("right_meters_per_count", &robot::DriveCalibration::rightMetersPerCount)
    .def_readwrite("left_sign", &robot::DriveCalibration::leftSign)
    .def_readwrite("right_sign", &robot::DriveCalibration::rightSign)
    .def_readwrite("left_feedforward", &robot::DriveCalibration::leftFeedforward)
    .def_readwrite("right_feedforward", &robot::DriveCalibration::rightFeedforward)
    .def_readwrite("left_deadband", &robot::DriveCalibration::leftDeadband)
    .def_readwrite("right_deadband", &robot::DriveCalibration::rightDeadband)
    .def_readwrite("kp", &robot::DriveCalibration::kp).def_readwrite("ki", &robot::DriveCalibration::ki)
    .def_readwrite("max_speed", &robot::DriveCalibration::maxSpeed);
  py::class_<robot::DriveController>(m, "Controller")
    .def(py::init<>())
    .def_readwrite("calibration", &robot::DriveController::calibration)
    .def("arm", &robot::DriveController::arm)
    .def("drive", &robot::DriveController::drive)
    .def("tick", &robot::DriveController::tick)
    .def("stop", [](robot::DriveController& c) { c.stop(robot::StopReason::Operator); })
    .def("set_model", [](robot::DriveController& c, std::array<float,4> mean,
        std::array<float,4> scale, std::array<std::array<float,4>,2> coefficients,
        std::array<float,2> intercept, float minSpeed, float maxSpeed, bool measured,
        float minCorrection, float maxCorrection) {
      auto& model = c.model;
      model.fitted = true; model.measured = measured;
      model.minSpeed = minSpeed; model.maxSpeed = maxSpeed;
      model.minCorrection = minCorrection; model.maxCorrection = maxCorrection;
      for (int i = 0; i < 4; ++i) {
        model.mean[i] = mean[i]; model.scale[i] = scale[i];
        for (int j = 0; j < 2; ++j) model.coefficients[j][i] = coefficients[j][i];
      }
      for (int j = 0; j < 2; ++j) model.intercept[j] = intercept[j];
      if (!model.valid()) throw py::value_error("Invalid correction model");
    })
    .def("choose", [](const robot::DriveController& c, float speed) {
      float delta;
      if (!c.model.choose(speed, delta)) throw py::value_error("Speed outside fitted model");
      return delta;
    })
    .def("predict", [](const robot::DriveController& c, float speed, float delta) {
      float y, heading; c.model.predict(speed, delta, y, heading);
      return std::array<float,2>{y, heading};
    })
    .def("set_student", [](robot::DriveController& c, py::dict w) {
      auto& s = c.student;
      s.fitted = true; s.measured = w["measured"].cast<bool>();
      s.minSpeed = w["min_speed"].cast<float>(); s.maxSpeed = w["max_speed"].cast<float>();
      s.minCorrection = w["min_correction"].cast<float>(); s.maxCorrection = w["max_correction"].cast<float>();
      s.trackM = w["track_m"].cast<float>(); s.elapsedCap = w["elapsed_cap_s"].cast<float>();
      auto mean = w["mean"].cast<std::vector<float>>(), scale = w["scale"].cast<std::vector<float>>();
      auto w1 = w["w1"].cast<std::vector<std::vector<float>>>(), w2 = w["w2"].cast<std::vector<std::vector<float>>>();
      auto b1 = w["b1"].cast<std::vector<float>>(), b2 = w["b2"].cast<std::vector<float>>();
      auto w3 = w["w3"].cast<std::vector<float>>();
      constexpr int I = robot::StudentModel::INPUTS, H = robot::StudentModel::HIDDEN;
      if (mean.size() != I || scale.size() != I || w1.size() != H || w2.size() != H ||
          b1.size() != H || b2.size() != H || w3.size() != H) throw py::value_error("Student shape");
      for (int i = 0; i < I; ++i) { s.mean[i] = mean[i]; s.scale[i] = scale[i]; }
      for (int j = 0; j < H; ++j) {
        if (w1[j].size() != I || w2[j].size() != H) throw py::value_error("Student shape");
        for (int i = 0; i < I; ++i) s.w1[j][i] = w1[j][i];
        for (int i = 0; i < H; ++i) s.w2[j][i] = w2[j][i];
        s.b1[j] = b1[j]; s.b2[j] = b2[j]; s.w3[j] = w3[j];
      }
      s.b3 = w["b3"].cast<float>();
      if (!s.valid()) throw py::value_error("Invalid student model");
    })
    .def("student_predict", [](const robot::DriveController& c, std::array<float, 5> input) {
      return c.student.predict(input.data());
    })
    .def("state", [](const robot::DriveController& c) {
      py::dict result;
      result["armed"] = c.armed; result["epoch"] = c.epoch; result["sequence"] = c.sequence;
      result["left"] = c.left; result["right"] = c.right; result["servo_us"] = c.servo;
      result["target_left"] = c.targetLeft; result["target_right"] = c.targetRight;
      result["pwm_left"] = robot::motorDuty(c.left); result["pwm_right"] = robot::motorDuty(c.right);
      result["reason"] = robot::reasonName(c.reason);
      result["speed_mps"] = c.speed; result["correction"] = c.correction;
      result["measured_left_mps"] = c.measuredLeft; result["measured_right_mps"] = c.measuredRight;
      return result;
    });
}
