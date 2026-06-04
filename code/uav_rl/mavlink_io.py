"""MAVLink command helpers for ArduPilot fixed-wing training."""

import time

import pymavlink.dialects.v20.common as mavlink2
from pymavlink import mavutil


def send_guided_goto(master, lat_deg: float, lon_deg: float, alt_rel_m: float) -> None:
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    master.mav.mission_item_int_send(
        master.target_system,
        master.target_component,
        0,
        frame,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        2,
        0,
        0,
        0,
        0,
        0,
        int(lat_deg * 1e7),
        int(lon_deg * 1e7),
        float(alt_rel_m),
    )


def send_guided_change_speed(master, groundspeed_mps: float) -> None:
    master.mav.command_int_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43000,
        0,
        0,
        1,
        float(groundspeed_mps),
        0.0,
        0,
        0,
        0,
        0,
    )


def send_guided_change_heading(master, heading_deg: float, max_centrip_acc: float = 2.0) -> None:
    master.mav.command_int_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_MISSION,
        43002,
        0,
        0,
        0,
        float(heading_deg),
        float(max_centrip_acc),
        0,
        0,
        0,
        0,
    )


def send_guided_change_alt(master, target_alt_m: float, rate_mps: float = 2.0) -> None:
    master.mav.command_int_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        43001,
        0,
        0,
        0,
        0,
        float(rate_mps),
        0,
        0,
        0,
        float(target_alt_m),
    )


def set_param(master, name: str, value: float) -> None:
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )


def wait_heartbeat(master, timeout_s: float = 30.0):
    heartbeat = master.wait_heartbeat(timeout=timeout_s)
    if heartbeat is None:
        raise TimeoutError(f"No MAVLink heartbeat received within {timeout_s} s")
    return heartbeat


def set_mode(master, mode: str) -> None:
    mode_mapping = master.mode_mapping()
    if mode not in mode_mapping:
        raise ValueError(f"Flight mode '{mode}' is not supported by this vehicle")
    try:
        master.set_mode(mode)
    except Exception:
        master.mav.set_mode_send(
            master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_mapping[mode],
        )


def wait_mode(master, mode: str, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg is None:
            continue
        current = mavutil.mode_string_v10(msg)
        if current == mode:
            return
    raise TimeoutError(f"Vehicle did not enter mode '{mode}' within {timeout_s} s")


def arm_vehicle(master) -> None:
    send_command_long(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)


def wait_armed(master, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if master.motors_armed():
            return
        master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
    raise TimeoutError(f"Vehicle did not arm within {timeout_s} s")


def disarm_vehicle(master) -> None:
    send_command_long(master, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0.0)


def rc_override(master, channel_values: dict[int, int]) -> None:
    values = [65535] * 18
    for channel, pwm in channel_values.items():
        if not 1 <= channel <= 18:
            raise ValueError(f"RC channel out of range: {channel}")
        values[channel - 1] = int(pwm)
    master.mav.rc_channels_override_send(
        master.target_system,
        master.target_component,
        *values[:8],
        *values[8:18],
    )


def clear_rc_override(master) -> None:
    rc_override(master, {idx: 0 for idx in range(1, 19)})


def send_command_long(master, command: int, *params: float) -> None:
    padded = list(params[:7]) + [0.0] * max(0, 7 - len(params))
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        command,
        0,
        *padded[:7],
    )


def request_message_interval(master, message_id: int, hz: float) -> None:
    interval_us = 0 if hz <= 0.0 else int(1e6 / hz)
    send_command_long(
        master,
        mavlink2.MAV_CMD_SET_MESSAGE_INTERVAL,
        float(message_id),
        float(interval_us),
    )
