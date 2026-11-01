#pragma once
#include "drive.h"
// Generated from measured runs; camera pose is never an inference input.
inline robot::CorrectionModel deployedModel() {
  robot::CorrectionModel m;
  m.fitted = true; m.measured = true;
  m.minSpeed = 1.116088886e-01f; m.maxSpeed = 1.984158019e-01f;
  m.minCorrection = -2.000000000e-02f; m.maxCorrection = 2.000000000e-02f;
  m.mean[0] = 1.535156743e-01f;
  m.mean[1] = -3.448275862e-04f;
  m.mean[2] = -6.841924205e-05f;
  m.mean[3] = 2.034482759e-04f;
  m.scale[0] = 4.337764433e-02f;
  m.scale[1] = 1.425936078e-02f;
  m.scale[2] = 2.284909157e-03f;
  m.scale[3] = 1.691413582e-04f;
  m.coefficients[0][0] = -1.785065462e-02f;
  m.coefficients[0][1] = -9.583450221e-02f;
  m.coefficients[0][2] = 6.708247057e-02f;
  m.coefficients[0][3] = 6.755534688e-03f;
  m.intercept[0] = 2.003327557e-02f;
  m.coefficients[1][0] = -4.029793942e-02f;
  m.coefficients[1][1] = -2.619220646e-01f;
  m.coefficients[1][2] = 1.517786289e-01f;
  m.coefficients[1][3] = 1.879890203e-02f;
  m.intercept[1] = -3.807633265e-02f;
  return m;
}
inline const char* deployedModelId() { return "e1dd0814ec65629c"; }
