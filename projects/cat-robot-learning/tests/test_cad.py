import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from catrobot.cad import ASSET_ROOT, convert_step, inspection_html, load_bundle, read_obj
from catrobot.config import RobotConfig, load_config, write_json
from catrobot.simulation import robot_xml, run_episode
from catrobot.submission import package_submission

BUNDLE = ASSET_ROOT / "shapr3d-20261029"


def visual_config():
    return load_config(Path(__file__).resolve().parents[1] / "examples/cad-visual-config.json")


def test_shapr3d_geometry_keeps_units_parts_and_positions():
    manifest, meshes = load_bundle(BUNDLE)
    assert manifest["source_length_units"] == ["millimetre"]
    assert manifest["mesh_length_unit"] == "metre"
    assert manifest["dimensions_m"] == pytest.approx([0.15, 0.15, 0.049])
    assert [solid["step_name"] for solid in manifest["solids"]] == ["ボディ 03", "ボディ 02", "ボディ 01"]
    assert manifest["mass_kg"] is None and manifest["hardware_alignment"] == "unconfirmed"
    for solid in manifest["solids"]:
        vertices, triangles = read_obj(meshes[solid["file"]])
        assert len(triangles) == solid["triangles"]
        bounds = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])
        assert np.allclose(bounds, solid["bounds_m"], atol=0.00015)
    source = (BUNDLE / "chassis.step").read_text()
    header = source.split("DATA;", 1)[0]
    assert "FILE_NAME('chassis.step','',(''),('')" in header
    assert "/Users/" not in source


def test_step_reconversion_is_reproducible(tmp_path):
    pytest.importorskip("OCP", reason="Optional STEP conversion group is not installed")
    expected, expected_meshes = load_bundle(BUNDLE)
    actual = convert_step(BUNDLE / "chassis.step", tmp_path / "converted")
    assert actual["source_sha256"] == expected["source_sha256"]
    assert actual["dimensions_m"] == pytest.approx(expected["dimensions_m"])
    _, actual_meshes = load_bundle(tmp_path / "converted")
    assert actual_meshes == expected_meshes
    with pytest.raises(ValueError, match="already exists"):
        convert_step(BUNDLE / "chassis.step", tmp_path / "converted")


def test_cad_visuals_do_not_change_mass_or_driving_physics(tmp_path):
    cad = visual_config()
    primitive = cad.model_copy(update={"cad_visual": None})
    for name, config in (("primitive", primitive), ("cad", cad)):
        run_episode(config, tmp_path / name, max_seconds=0.8)
    worlds = [
        mujoco.MjModel.from_xml_path(str(tmp_path / name / "robot.xml"))
        for name in ("primitive", "cad")
    ]
    assert worlds[1].nmesh == 3
    assert np.array_equal(worlds[0].body_mass, worlds[1].body_mass)
    assert np.array_equal(worlds[0].body_inertia, worlds[1].body_inertia)
    assert (tmp_path / "primitive/poses.jsonl").read_bytes() == (tmp_path / "cad/poses.jsonl").read_bytes()
    for i in range(3):
        geom = mujoco.mj_name2id(worlds[1], mujoco.mjtObj.mjOBJ_GEOM, f"cad_solid_{i}")
        assert worlds[1].geom_contype[geom] == worlds[1].geom_conaffinity[geom] == 0
    metadata = json.loads((tmp_path / "cad/metadata.json").read_text())
    assert metadata["cad_visual"]["alignment"] == "orientation_confirmed"
    assert metadata["ball_caster_count"] == 2
    assert metadata["collision_model"] == "primitive_proxy"
    path = tmp_path / "config.json"
    write_json(path, cad.model_dump())
    assert load_config(path).fingerprint() == cad.fingerprint()
    assert cad.fingerprint() != primitive.fingerprint()


def test_cad_front_and_passive_supports_match_reported_assembly(tmp_path):
    from catrobot.cad import visual_bundle

    config = visual_config()
    _, assets = visual_bundle(config.cad_visual)
    world = mujoco.MjModel.from_xml_string(robot_xml(config), assets)
    state = mujoco.MjData(world)
    mujoco.mj_forward(world, state)
    quaternion = np.array(config.cad_visual.quaternion_wxyz)
    forward, left = np.zeros(3), np.zeros(3)
    mujoco.mju_rotVecQuat(forward, np.array([0.0, 1.0, 0.0]), quaternion)
    mujoco.mju_rotVecQuat(left, np.array([-1.0, 0.0, 0.0]), quaternion)
    assert np.allclose(forward, [1, 0, 0], atol=1e-12)
    assert np.allclose(left, [0, 1, 0], atol=1e-12)
    chassis = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    for i, expected_x in enumerate((0.05, -0.05)):
        body = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_BODY, f"caster_{i}")
        joint = world.body_jntadr[body]
        assert world.jnt_type[joint] == mujoco.mjtJoint.mjJNT_BALL
        assert np.allclose(state.xpos[body] - state.xpos[chassis], [expected_x, 0, -0.0415])
    right_cap = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_GEOM, "cad_solid_0")
    left_cap = mujoco.mj_name2id(world, mujoco.mjtObj.mjOBJ_GEOM, "cad_solid_1")
    assert state.geom_xpos[right_cap, 1] < 0 < state.geom_xpos[left_cap, 1]
    result = run_episode(config, tmp_path / "assembled", max_seconds=2)
    assert result["forward_m"] > 0.005
    assert result["ball_caster_count"] == 2
    assert not result["physics_calibrated"]
    ball_plane = config.ball_casters[0].position_m[2] - config.ball_casters[0].radius_m
    assert ball_plane == pytest.approx(-0.03 - config.left.radius_m)


def test_pre_cad_model_identity_stays_compatible():
    assert RobotConfig().fingerprint() == "f977699bfa1f1fcbc177fae58b6720fcb3fb7368e7b27b5ba7a12dfe799e06ee"
    assert len(RobotConfig().resolved_ball_casters()) == 1


def test_review_snapshot_contains_self_contained_cad_assets(tmp_path):
    project = Path(__file__).resolve().parents[1]
    result = package_submission(project, tmp_path / "snapshot")
    assert not result["published"]
    copied = tmp_path / "snapshot/projects/cat-robot-learning/src/catrobot"
    manifest, meshes = load_bundle(copied / "assets/shapr3d-20261029")
    assert len(meshes) == 3 and (copied / "cad_view.html").exists()
    html = inspection_html(copied / "assets/shapr3d-20261029")
    assert "__CAD_DATA__" not in html
    assert "DecompressionStream" in html and "https://" not in html
    assert manifest["mass_kg"] is None
    project_copy = tmp_path / "snapshot/projects/cat-robot-learning"
    assert (project_copy / "foundation/teacher.py").exists()
    assert (project_copy / "report/final-report.md").exists()
