# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np

from omnisurg.hex.haptic import (
    MINIMOU_PROFILE,
    MINIMOU_PITCH_FIX_DEGREES,
    OPENHAPTICS_PROFILE,
    HapticFrameConfig,
    InputPose,
    MiniMouInput,
    axis_degrees_to_quaternion,
    minimou_angles_to_quaternion,
    minimou_orientation_to_quaternion,
    minimou_position_to_adapter,
    minimou_reflect_x_handedness,
    pose_to_world,
    quat_mul,
    quat_to_matrix,
    scene_basis,
    y_up_pitch_yaw_correction,
)


def _assert_matrix_close(q, expected_q, atol: float = 1.0e-6) -> None:
    assert np.allclose(quat_to_matrix(q), quat_to_matrix(expected_q), atol=atol)


def _config(profile, *, scale: float = 1.0, up_axis: str = "Z") -> HapticFrameConfig:
    return HapticFrameConfig(profile=profile, position_scale=scale, up_axis=up_axis)


def test_scene_basis_axes_are_deterministic_and_orthonormal():
    expected = {
        "Z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "Y": ((1.0, 0.0, 0.0), (-0.0, -0.0, -1.0), (0.0, 1.0, 0.0)),
        # Current X-up convention: right=Y, depth=Z, up=X.
        "X": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    }
    for up_axis, expected_basis in expected.items():
        basis = scene_basis(up_axis)
        assert all(np.allclose(axis, expected_axis) for axis, expected_axis in zip(basis, expected_basis, strict=True))
        basis_matrix = np.column_stack(basis)
        assert np.allclose(basis_matrix.T @ basis_matrix, np.eye(3), atol=1.0e-7)


def test_openhaptics_z_up_identity_maps_to_identity():
    world = pose_to_world(InputPose(valid=True), _config(OPENHAPTICS_PROFILE))
    _assert_matrix_close(world.quaternion, (0.0, 0.0, 0.0, 1.0))


def test_openhaptics_z_up_pure_rotations_match_current_mapping():
    cases = [
        ((1.0, 0.0, 0.0), 30.0, (1.0, 0.0, 0.0), -30.0),
        ((0.0, 1.0, 0.0), 30.0, (0.0, 0.0, 1.0), -30.0),
        ((0.0, 0.0, 1.0), 30.0, (0.0, 1.0, 0.0), 30.0),
    ]
    for raw_axis, raw_angle, world_axis, world_angle in cases:
        raw_q = axis_degrees_to_quaternion(*raw_axis, raw_angle)
        world = pose_to_world(InputPose(quaternion=raw_q, valid=True), _config(OPENHAPTICS_PROFILE))
        expected_q = axis_degrees_to_quaternion(*world_axis, world_angle)
        _assert_matrix_close(world.quaternion, expected_q)


def test_openhaptics_y_up_pitch_and_yaw_are_not_inverted():
    cases = [
        ((1.0, 0.0, 0.0), 30.0, (1.0, 0.0, 0.0), 30.0),
        ((0.0, 1.0, 0.0), 30.0, (0.0, 1.0, 0.0), 30.0),
        ((0.0, 0.0, 1.0), 30.0, (0.0, 0.0, 1.0), -30.0),
    ]
    for raw_axis, raw_angle, world_axis, world_angle in cases:
        raw_q = axis_degrees_to_quaternion(*raw_axis, raw_angle)
        world = pose_to_world(InputPose(quaternion=raw_q, valid=True), _config(OPENHAPTICS_PROFILE, up_axis="Y"))
        expected_q = axis_degrees_to_quaternion(*world_axis, world_angle)
        _assert_matrix_close(world.quaternion, expected_q)


def test_openhaptics_position_preserves_current_mapping():
    config = HapticFrameConfig(
        profile=OPENHAPTICS_PROFILE,
        position_scale=1.0,
        center=(1.0, 2.0, 3.0),
        device_offset=(0.1, 0.2, 0.3),
        up_axis="Z",
    )
    world = pose_to_world(InputPose(position=(10.0, 20.0, 30.0), valid=True), config)
    assert np.allclose(world.position, (11.1, -27.8, 23.3))


class _FakeMiniMouController:
    def __init__(self) -> None:
        self.closed = False

    def perform_update(self) -> None:
        pass

    def get_position(self):
        return (1.0, 2.0, 3.0)

    def get_rot_angle(self) -> float:
        return 10.0

    def get_pitch_angle(self) -> float:
        return 20.0

    def get_yaw_angle(self) -> float:
        return 30.0

    def get_orientation(self):
        return (0.0, 0.0, 1.0, math.radians(30.0))

    def close(self) -> None:
        self.closed = True


class _FakeMiniMouHandleController(_FakeMiniMouController):
    def __init__(self) -> None:
        super().__init__()
        self.handle_opening = 1.0
        self.handle_activity = 0

    def get_tool_pos(self) -> float:
        return 1.0

    def get_handle_opening_value(self) -> float:
        return self.handle_opening

    def get_handle_activity(self) -> int:
        return self.handle_activity


def test_minimou_poll_uses_adapter_position_and_axis_angle_orientation():
    controller = _FakeMiniMouController()
    device = MiniMouInput(controller)

    pose = device.poll()

    assert pose.valid
    assert pose.position == (-1.0, 2.0, 3.0)
    assert device.angles_degrees() == (10.0, 20.0, 30.0)
    _assert_matrix_close(pose.quaternion, minimou_orientation_to_quaternion(controller.get_orientation()))


def test_minimou_poll_uses_physical_handle_for_grip_trigger():
    controller = _FakeMiniMouHandleController()
    device = MiniMouInput(controller)

    open_pose = device.poll()
    controller.handle_opening = 0.0
    controller.handle_activity = 1
    squeezed_pose = device.poll()

    assert open_pose.valid
    assert not open_pose.button1
    assert open_pose.handle_pos == 1.0
    assert open_pose.grip == 0.0
    assert squeezed_pose.button1
    assert squeezed_pose.handle_active
    assert squeezed_pose.handle_pos == 0.0
    assert squeezed_pose.grip == 1.0


def test_minimou_discover_uses_repo_local_follou_import_helper(monkeypatch):
    from omnisurg.input import follou as follou_module

    calls = []

    class FakeMiniMou:
        pass

    class FakeManager:
        def get_device_controller(self, device_cls, idx):
            calls.append((device_cls, idx))
            return _FakeMiniMouController()

    monkeypatch.setattr(follou_module, "ensure_follou_importable", lambda: (FakeManager, FakeMiniMou))

    devices = MiniMouInput.discover(count=1)

    assert len(devices) == 1
    assert calls == [(FakeMiniMou, 0)]
    devices[0].close()


def test_minimou_adapter_position_preserves_current_mapping():
    assert minimou_position_to_adapter((1.0, 2.0, 3.0)) == (-1.0, 2.0, 3.0)


def test_minimou_z_up_adapter_position_preserves_current_world_mapping():
    pose = InputPose(position=minimou_position_to_adapter((1.0, 2.0, 3.0)), valid=True)
    world = pose_to_world(pose, _config(MINIMOU_PROFILE))
    assert np.allclose(world.position, (-1.0, -3.0, 2.0))


def test_minimou_reflect_x_handedness_is_named_and_tested():
    reflected = minimou_reflect_x_handedness(axis_degrees_to_quaternion(0.0, 1.0, 0.0, 30.0))
    _assert_matrix_close(reflected, axis_degrees_to_quaternion(0.0, 1.0, 0.0, -30.0))


def test_minimou_orientation_axis_angle_converts_to_right_handed_adapter():
    _assert_matrix_close(
        minimou_orientation_to_quaternion((0.0, 0.0, 0.0, math.radians(30.0))),
        (0.0, 0.0, 0.0, 1.0),
    )
    # The firmware-to-adapter basis change is a reflection (det = -1), so the
    # axis is a pseudovector: under the X-reflection, the (Y, Z) components of
    # the axis flip sign. Equivalently, a rotation around firmware X reads as
    # the opposite-signed rotation around adapter X, while rotations around
    # firmware Y/Z keep their signed angle in the adapter Y/Z components.
    cases = [
        ((1.0, 0.0, 0.0, math.radians(30.0)), (1.0, 0.0, 0.0, -30.0)),
        ((0.0, 1.0, 0.0, math.radians(30.0)), (0.0, 1.0, 0.0, 30.0)),
        ((0.0, 0.0, 1.0, math.radians(30.0)), (0.0, 0.0, 1.0, 30.0)),
    ]
    for orientation, expected_axis_angle in cases:
        _assert_matrix_close(
            minimou_orientation_to_quaternion(orientation),
            axis_degrees_to_quaternion(*expected_axis_angle),
        )


def test_minimou_z_up_pure_joint_angles_match_world_axes():
    pitch_fix_q = axis_degrees_to_quaternion(1.0, 0.0, 0.0, MINIMOU_PITCH_FIX_DEGREES)
    cases = [
        (
            (30.0, 0.0, 0.0),
            minimou_reflect_x_handedness(quat_mul(axis_degrees_to_quaternion(0.0, 1.0, 0.0, -30.0), pitch_fix_q)),
        ),
        (
            (0.0, 30.0, 0.0),
            axis_degrees_to_quaternion(1.0, 0.0, 0.0, MINIMOU_PITCH_FIX_DEGREES - 30.0),
        ),
        (
            (0.0, 0.0, 30.0),
            minimou_reflect_x_handedness(quat_mul(axis_degrees_to_quaternion(0.0, 0.0, 1.0, -30.0), pitch_fix_q)),
        ),
    ]
    for angles, expected_q in cases:
        pose = InputPose(quaternion=minimou_angles_to_quaternion(*angles), valid=True)
        world = pose_to_world(pose, _config(MINIMOU_PROFILE))
        _assert_matrix_close(world.quaternion, expected_q)


def test_minimou_z_up_zero_angles_apply_fixed_pitch_calibration():
    pose = InputPose(quaternion=minimou_angles_to_quaternion(0.0, 0.0, 0.0), valid=True)
    world = pose_to_world(pose, _config(MINIMOU_PROFILE))
    _assert_matrix_close(
        world.quaternion,
        axis_degrees_to_quaternion(1.0, 0.0, 0.0, MINIMOU_PITCH_FIX_DEGREES),
    )


def test_minimou_z_up_roll_and_yaw_do_not_cancel():
    neutral = minimou_angles_to_quaternion(0.0, 0.0, 0.0)
    combined = minimou_angles_to_quaternion(30.0, 0.0, 30.0)

    assert not np.allclose(quat_to_matrix(combined), quat_to_matrix(neutral), atol=1.0e-6)


def test_minimou_y_up_pitch_and_yaw_are_not_inverted():
    pitch_fix_q = axis_degrees_to_quaternion(1.0, 0.0, 0.0, MINIMOU_PITCH_FIX_DEGREES)
    cases = [
        (
            (30.0, 0.0, 0.0),
            y_up_pitch_yaw_correction(
                minimou_reflect_x_handedness(
                    quat_mul(axis_degrees_to_quaternion(0.0, 1.0, 0.0, -30.0), pitch_fix_q)
                )
            ),
        ),
        (
            (0.0, 30.0, 0.0),
            y_up_pitch_yaw_correction(
                axis_degrees_to_quaternion(1.0, 0.0, 0.0, MINIMOU_PITCH_FIX_DEGREES - 30.0)
            ),
        ),
        (
            (0.0, 0.0, 30.0),
            y_up_pitch_yaw_correction(
                minimou_reflect_x_handedness(
                    quat_mul(axis_degrees_to_quaternion(0.0, 0.0, 1.0, -30.0), pitch_fix_q)
                )
            ),
        ),
    ]
    for angles, expected_q in cases:
        pose = InputPose(quaternion=minimou_angles_to_quaternion(*angles), valid=True)
        world = pose_to_world(pose, _config(MINIMOU_PROFILE, up_axis="Y"))
        _assert_matrix_close(world.quaternion, expected_q)
