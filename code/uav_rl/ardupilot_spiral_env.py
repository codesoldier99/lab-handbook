"""Online ArduPilot spiral-training environment with optional Gazebo launch."""

from __future__ import annotations

import math
import os
import random
import re
import signal
import shlex
import subprocess
import time
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from pymavlink import mavutil

from .mavlink_io import (
    arm_vehicle,
    clear_rc_override,
    rc_override,
    request_message_interval,
    send_guided_change_alt,
    send_guided_change_heading,
    send_guided_change_speed,
    send_guided_goto,
    set_mode,
    set_param,
    wait_heartbeat,
    wait_mode,
)
from .spiral_geometry import (
    clamp,
    cross_track_error,
    latlon_to_ne,
    ne_error_from_latlon,
    ne_to_latlon,
    shift_latlon,
    wrap_360,
    wrap_pi,
)


@dataclass
class SpiralEnvConfig:
    master: str = "udp:127.0.0.1:14550"
    auto_launch: bool = False
    relaunch_every_episode: bool = True
    launch_gazebo: bool = False
    keep_gazebo_alive: bool = False
    gazebo_cmd: str = "gz sim -v4 -r zephyr_runway_nowind.sdf --render-engine ogre"
    gazebo_cwd: str = ""
    gazebo_ready_sleep_s: float = 12.0
    gazebo_world_name: str = "default"
    gazebo_model_name: str = "zephyr"
    gazebo_base_pose: str = ""
    gazebo_base_pose_degrees: bool = True
    gazebo_random_x_m: float = 0.0
    gazebo_random_y_m: float = 0.0
    gazebo_random_xy_radius_m: float = 0.0
    gazebo_random_z_m: float = 0.0
    gazebo_random_yaw_deg: float = 0.0
    gazebo_transport_timeout_ms: int = 5000
    gazebo_real_time_factor: float = 1.0
    gazebo_reset_mode: str = "time"
    gazebo_reset_sleep_s: float = 1.0
    airborne_reset_after_first_episode: bool = False
    airborne_reset_pose: str = ""
    airborne_reset_alt_m: float = 30.0
    airborne_reset_xy_radius_m: float = -1.0
    airborne_reset_yaw_deg: float = -1.0
    airborne_reset_settle_s: float = 2.0
    ardupilot_cmd: str = "sim_vehicle.py -v ArduPlane -f gazebo-zephyr --model JSON --out=udp:127.0.0.1:14550"
    ardupilot_cwd: str = ""
    ardupilot_ready_sleep_s: float = 20.0
    home_base: str = ""
    home_random_radius_m: float = 0.0
    home_random_alt_m: float = 0.0
    home_random_heading_deg: float = 0.0
    heartbeat_timeout_s: float = 45.0
    ready_timeout_s: float = 240.0
    step_wall_timeout_s: float = 2.0
    teardown_sleep_s: float = 3.0
    use_sim_time_control: bool = True

    takeoff_mode: str = "FBWA"
    climb_mode: str = "GUIDED"
    skip_takeoff: bool = False
    takeoff_throttle_channel: int = 3
    takeoff_throttle_pwm: int = 1800
    takeoff_release_pwm: int = 1500
    arm_wait_s: float = 15.0
    prearm_settle_s: float = 8.0
    arm_retry_interval_s: float = 2.0
    takeoff_throttle_refresh_s: float = 0.2
    spiral_start_alt: float = 100.0
    wind_enable_alt: float = 20.0
    takeoff_level_alt_m: float = 10.0
    takeoff_loiter_dist_m: float = 120.0
    takeoff_thr_max_pct: float = 100.0
    takeoff_thr_minacc: float = 0.0
    takeoff_thr_minspd: float = 0.0
    takeoff_thr_delay_s: float = 0.0

    turns: float = 3.0
    r0: float = 80.0
    r_step: float = 80.0
    direction: str = "CW"
    start_radial_deg: float = 90.0
    lookahead_deg: float = 15.0

    speed: float = 15.0
    cmd_hz: float = 5.0
    centrip_acc: float = 2.0
    spiral_climb: float = 200.0
    climb_rate: float = 2.0

    wind_from_deg: float = 180.0
    wind_speed_mps: float = 0.0
    wind_turb: float = 0.0
    wind_ramp_s: float = 0.0

    k_min: float = 0.5
    k_max: float = 2.0
    k_alt_min: float = 0.5
    k_alt_max: float = 2.0
    action_space: str = "heading"
    action_mode: str = "absolute_k"
    command_mode: str = "full_guided"
    goto_every_steps: int = 1
    l_corr: float = 50.0
    max_corr_deg: float = 25.0
    max_alt_corr_m: float = 15.0

    expected_radial_error_m: float = 5.0
    expected_alt_error_m: float = 5.0
    terminate_radial_error_m: float = 80.0
    min_safe_alt_m: float = 20.0
    max_episode_steps: int = 5000

    reward_mode: str = "continuous_error"
    reward_radial_weight: float = 1.0
    reward_alt_weight: float = 0.5
    reward_cross_track_weight: float = 0.2
    reward_heading_weight: float = 0.05
    reward_smooth_weight: float = 0.02
    reward_progress_weight: float = 0.4
    reward_step_penalty: float = 0.01
    reward_completion_bonus: float = 0.0
    reward_in_band_bonus: float = 2.0
    reward_failure_penalty: float = 50.0


def config_from_env(prefix: str = "UAV_RL_") -> SpiralEnvConfig:
    kwargs = {}
    for item in fields(SpiralEnvConfig):
        env_name = f"{prefix}{item.name}".upper()
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        default_type = type(item.default)
        if default_type is bool:
            kwargs[item.name] = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif default_type is int:
            kwargs[item.name] = int(raw)
        elif default_type is float:
            kwargs[item.name] = float(raw)
        else:
            kwargs[item.name] = raw
    return SpiralEnvConfig(**kwargs)


def parse_home_spec(home_spec: str) -> tuple[float, float, float, float]:
    parts = [item.strip() for item in home_spec.split(",")]
    if len(parts) != 4:
        raise ValueError("home_base must use 'lat,lon,alt,heading'")
    lat_deg, lon_deg, alt_m, heading_deg = [float(item) for item in parts]
    return lat_deg, lon_deg, alt_m, heading_deg


def format_home_spec(lat_deg: float, lon_deg: float, alt_m: float, heading_deg: float) -> str:
    return f"{lat_deg:.7f},{lon_deg:.7f},{alt_m:.2f},{heading_deg:.2f}"


def parse_gazebo_pose_spec(pose_spec: str, degrees: bool = True) -> tuple[float, float, float, float, float, float]:
    parts = [item.strip() for item in pose_spec.split(",")]
    if len(parts) != 6:
        raise ValueError("gazebo_base_pose must use 'x,y,z,roll,pitch,yaw'")
    x_m, y_m, z_m, roll_value, pitch_value, yaw_value = [float(item) for item in parts]
    if degrees:
        roll_value = math.radians(roll_value)
        pitch_value = math.radians(pitch_value)
        yaw_value = math.radians(yaw_value)
    return x_m, y_m, z_m, roll_value, pitch_value, yaw_value


def inject_home_into_command(command: str, home_spec: str) -> str:
    custom_location_pattern = re.compile(r"(--custom-location(?:=|\s+))(\S+)")
    short_location_pattern = re.compile(r"(-l(?:\s+))(\S+)")
    named_location_pattern = re.compile(r"(-L(?:\s+)|--location(?:=|\s+))(\S+)")
    if custom_location_pattern.search(command):
        return custom_location_pattern.sub(rf"\1{home_spec}", command, count=1)
    if short_location_pattern.search(command):
        return short_location_pattern.sub(rf"\1{home_spec}", command, count=1)
    if named_location_pattern.search(command):
        command = named_location_pattern.sub("", command, count=1).strip()
    return f"{command} --custom-location={home_spec}"


class SimulatorProcessGroup:
    def __init__(self, cfg: SpiralEnvConfig):
        self.cfg = cfg
        self.gazebo_proc: subprocess.Popen | None = None
        self.ardupilot_proc: subprocess.Popen | None = None
        self.last_home_spec: str | None = None
        self.generated_world_path: Path | None = None

    def start(self) -> None:
        self._validate_launch_configuration()
        self.stop()
        self._start_gazebo()
        if self.cfg.launch_gazebo:
            self._set_gazebo_pose_if_configured()
        self._start_ardupilot()

    def stop(self) -> None:
        self._stop_ardupilot()
        self._stop_gazebo()
        time.sleep(max(0.0, self.cfg.teardown_sleep_s))

    def prepare_episode(self) -> None:
        self._validate_launch_configuration()
        if self.cfg.launch_gazebo and self.cfg.keep_gazebo_alive:
            self._ensure_gazebo_running()
            self._stop_ardupilot()
            self._reset_gazebo_world()
            self._set_gazebo_pose_if_configured()
            self._start_ardupilot()
        else:
            self.start()

    def prepare_airborne_episode(self) -> None:
        self._validate_launch_configuration()
        if not (self.cfg.launch_gazebo and self.cfg.keep_gazebo_alive):
            raise RuntimeError("airborne reset requires launch_gazebo and keep_gazebo_alive.")
        self._ensure_gazebo_running()
        self._set_gazebo_paused(True)
        self._set_gazebo_pose(self._sample_airborne_pose())
        self._set_gazebo_paused(False)
        time.sleep(max(0.0, self.cfg.airborne_reset_settle_s))

    def _spawn(self, command: str, cwd: str) -> subprocess.Popen:
        if not command.strip():
            raise ValueError("Simulator launch command is empty")
        return subprocess.Popen(
            command,
            shell=True,
            cwd=str(Path(cwd).expanduser()) if cwd else None,
            start_new_session=True,
        )

    def _stop_ardupilot(self) -> None:
        self._terminate_process(self.ardupilot_proc)
        self.ardupilot_proc = None

    def _stop_gazebo(self) -> None:
        self._terminate_process(self.gazebo_proc)
        self.gazebo_proc = None

    def _terminate_process(self, proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()

    def _start_gazebo(self) -> None:
        if not self.cfg.launch_gazebo:
            self.gazebo_proc = None
            return
        self.gazebo_proc = self._spawn(self._build_gazebo_command(), self.cfg.gazebo_cwd)
        time.sleep(max(0.0, self.cfg.gazebo_ready_sleep_s))

    def _start_ardupilot(self) -> None:
        ardupilot_cmd = self._build_ardupilot_command()
        self.ardupilot_proc = self._spawn(ardupilot_cmd, self.cfg.ardupilot_cwd)
        time.sleep(max(0.0, self.cfg.ardupilot_ready_sleep_s))

    def _ensure_gazebo_running(self) -> None:
        if not self.cfg.launch_gazebo:
            return
        if self.gazebo_proc is None or self.gazebo_proc.poll() is not None:
            self._stop_gazebo()
            self._start_gazebo()

    def _build_ardupilot_command(self) -> str:
        command = self.cfg.ardupilot_cmd
        home_spec = self._sample_home_spec()
        self.last_home_spec = home_spec
        if home_spec:
            command = inject_home_into_command(command, home_spec)
            print(f"[SIM] launch home={home_spec}")
        return command

    def _build_gazebo_command(self) -> str:
        if self.cfg.gazebo_real_time_factor <= 0.0 or abs(self.cfg.gazebo_real_time_factor - 1.0) < 1e-6:
            return self.cfg.gazebo_cmd
        try:
            parts = shlex.split(self.cfg.gazebo_cmd)
        except ValueError:
            print("[GAZEBO] could not parse gazebo_cmd; using it without real_time_factor injection.")
            return self.cfg.gazebo_cmd
        world_index = self._find_world_arg_index(parts)
        if world_index is None:
            print("[GAZEBO] no .sdf/.world argument found; using gazebo_cmd without real_time_factor injection.")
            return self.cfg.gazebo_cmd
        world_path = self._resolve_world_path(parts[world_index])
        if world_path is None:
            print(
                f"[GAZEBO] could not resolve world [{parts[world_index]}]; "
                "using gazebo_cmd without real_time_factor injection."
            )
            return self.cfg.gazebo_cmd
        generated = self._write_speedup_world(world_path, self.cfg.gazebo_real_time_factor)
        parts[world_index] = str(generated)
        print(f"[GAZEBO] using temporary world {generated} with real_time_factor={self.cfg.gazebo_real_time_factor:g}")
        return shlex.join(parts)

    def _find_world_arg_index(self, parts: list[str]) -> int | None:
        for idx, part in enumerate(parts):
            lowered = part.lower()
            if lowered.endswith(".sdf") or lowered.endswith(".world"):
                return idx
        return None

    def _resolve_world_path(self, world_arg: str) -> Path | None:
        raw = Path(world_arg).expanduser()
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            cwd = Path(self.cfg.gazebo_cwd).expanduser() if self.cfg.gazebo_cwd else Path.cwd()
            candidates.append(cwd / raw)
            resource_vars = ["GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH", "GZ_FILE_PATH", "SDF_PATH"]
            for var_name in resource_vars:
                for root in os.environ.get(var_name, "").split(os.pathsep):
                    if not root:
                        continue
                    root_path = Path(root).expanduser()
                    candidates.append(root_path / raw)
                    candidates.append(root_path / "worlds" / raw.name)
            candidates.append(Path.home() / "gz_ws" / "src" / "ardupilot_gazebo" / "worlds" / raw.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _write_speedup_world(self, source: Path, real_time_factor: float) -> Path:
        text = source.read_text(encoding="utf-8")
        factor_text = f"{real_time_factor:g}"
        if re.search(r"<real_time_factor>.*?</real_time_factor>", text, flags=re.DOTALL):
            text = re.sub(
                r"<real_time_factor>.*?</real_time_factor>",
                f"<real_time_factor>{factor_text}</real_time_factor>",
                text,
                count=1,
                flags=re.DOTALL,
            )
        elif re.search(r"<max_step_size>.*?</max_step_size>", text, flags=re.DOTALL):
            text = re.sub(
                r"(<max_step_size>.*?</max_step_size>)",
                rf"\1\n      <real_time_factor>{factor_text}</real_time_factor>",
                text,
                count=1,
                flags=re.DOTALL,
            )
        else:
            updated = re.sub(
                r"(<physics[^>]*>)",
                rf"\1\n      <real_time_factor>{factor_text}</real_time_factor>",
                text,
                count=1,
            )
            if updated == text:
                raise RuntimeError(f"No <physics> block found in Gazebo world: {source}")
            text = updated
        out_dir = Path(tempfile.gettempdir()) / "uav_rl_gazebo_worlds"
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = out_dir / f"{source.stem}_rtf_{factor_text.replace('.', '_')}{source.suffix}"
        generated.write_text(text, encoding="utf-8")
        self.generated_world_path = generated
        return generated

    def _reset_gazebo_world(self) -> None:
        mode = self.cfg.gazebo_reset_mode.strip().lower()
        if mode in {"", "none", "off", "false"}:
            return
        reset_requests = {
            "all": "reset: {all: true}",
            "model": "reset: {model_only: true}",
            "models": "reset: {model_only: true}",
            "time": "reset: {time_only: true}",
            "time_only": "reset: {time_only: true}",
        }
        if mode not in reset_requests:
            raise ValueError("gazebo_reset_mode must be one of: none, time, model, all")
        self._run_gz_service(
            service=f"/world/{self.cfg.gazebo_world_name}/control",
            reqtype="gz.msgs.WorldControl",
            rep_type="gz.msgs.Boolean",
            request=reset_requests[mode],
        )
        time.sleep(max(0.0, self.cfg.gazebo_reset_sleep_s))

    def _set_gazebo_pose_if_configured(self) -> None:
        pose = self._sample_gazebo_pose()
        if pose is None:
            return
        self._set_gazebo_pose(pose)
        time.sleep(max(0.0, self.cfg.gazebo_reset_sleep_s))

    def _set_gazebo_pose(self, pose: tuple[float, float, float, float, float, float]) -> None:
        x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad = pose
        half_roll = roll_rad / 2.0
        half_pitch = pitch_rad / 2.0
        half_yaw = yaw_rad / 2.0
        cr = math.cos(half_roll)
        sr = math.sin(half_roll)
        cp = math.cos(half_pitch)
        sp = math.sin(half_pitch)
        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        req = (
            f'name: "{self.cfg.gazebo_model_name}" '
            f'position: {{x: {x_m:.4f} y: {y_m:.4f} z: {z_m:.4f}}} '
            f'orientation: {{x: {qx:.6f} y: {qy:.6f} z: {qz:.6f} w: {qw:.6f}}}'
        )
        self._run_gz_service(
            service=f"/world/{self.cfg.gazebo_world_name}/set_pose",
            reqtype="gz.msgs.Pose",
            rep_type="gz.msgs.Boolean",
            request=req,
        )
        print(
            f"[GAZEBO] set_pose model={self.cfg.gazebo_model_name} "
            f"x={x_m:.2f} y={y_m:.2f} z={z_m:.2f} yaw_deg={math.degrees(yaw_rad):.2f}"
        )

    def _set_gazebo_paused(self, paused: bool) -> None:
        self._run_gz_service(
            service=f"/world/{self.cfg.gazebo_world_name}/control",
            reqtype="gz.msgs.WorldControl",
            rep_type="gz.msgs.Boolean",
            request=f"pause: {str(paused).lower()}",
        )
        state = "paused" if paused else "running"
        print(f"[GAZEBO] simulation {state}")

    def _run_gz_service(self, service: str, reqtype: str, rep_type: str, request: str) -> None:
        cmd = [
            "gz",
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            rep_type,
            "--timeout",
            str(self.cfg.gazebo_transport_timeout_ms),
            "--req",
            request,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Gazebo service call failed for {service}: {result.stderr.strip() or result.stdout.strip()}"
            )

    def _validate_launch_configuration(self) -> None:
        cmd = self.cfg.ardupilot_cmd.lower()
        uses_json_model = "--model json" in cmd or "--model=json" in cmd
        uses_gazebo_zephyr = "-f gazebo-zephyr" in cmd or "--frame=gazebo-zephyr" in cmd
        if self.cfg.keep_gazebo_alive and not self.cfg.launch_gazebo:
            raise RuntimeError("keep_gazebo_alive requires launch_gazebo to be enabled.")
        if self.cfg.gazebo_base_pose.strip() == "" and (
            self.cfg.gazebo_random_x_m > 0.0
            or self.cfg.gazebo_random_y_m > 0.0
            or self.cfg.gazebo_random_xy_radius_m > 0.0
            or self.cfg.gazebo_random_z_m > 0.0
            or self.cfg.gazebo_random_yaw_deg > 0.0
        ):
            raise RuntimeError(
                "gazebo random pose options require gazebo_base_pose in the form 'x,y,z,roll,pitch,yaw'."
            )
        if self.cfg.auto_launch and (uses_json_model or uses_gazebo_zephyr) and not self.cfg.launch_gazebo:
            raise RuntimeError(
                "The selected ArduPilot command uses gazebo-zephyr/JSON, which requires an external JSON simulator "
                "such as Gazebo. Either enable --launch-gazebo, or switch --ardupilot-cmd to a non-JSON ArduPlane "
                "frame for ArduPilot-only training."
            )

    def _sample_home_spec(self) -> str | None:
        if not self.cfg.home_base.strip():
            return None
        lat_deg, lon_deg, alt_m, heading_deg = parse_home_spec(self.cfg.home_base)
        north_m = east_m = 0.0
        if self.cfg.home_random_radius_m > 0.0:
            angle = random.uniform(0.0, 2.0 * math.pi)
            radius = self.cfg.home_random_radius_m * math.sqrt(random.random())
            north_m = radius * math.cos(angle)
            east_m = radius * math.sin(angle)
        lat_deg, lon_deg = shift_latlon(lat_deg, lon_deg, north_m, east_m)
        if self.cfg.home_random_alt_m > 0.0:
            alt_m += random.uniform(-self.cfg.home_random_alt_m, self.cfg.home_random_alt_m)
        alt_m = max(0.0, alt_m)
        if self.cfg.home_random_heading_deg > 0.0:
            heading_deg += random.uniform(-self.cfg.home_random_heading_deg, self.cfg.home_random_heading_deg)
        return format_home_spec(lat_deg, lon_deg, alt_m, wrap_360(heading_deg))

    def _sample_gazebo_pose(self) -> tuple[float, float, float, float, float, float] | None:
        if not self.cfg.gazebo_base_pose.strip():
            return None
        x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad = parse_gazebo_pose_spec(
            self.cfg.gazebo_base_pose,
            degrees=self.cfg.gazebo_base_pose_degrees,
        )
        random_x_m = max(0.0, self.cfg.gazebo_random_x_m)
        random_y_m = max(0.0, self.cfg.gazebo_random_y_m)
        if random_x_m > 0.0 or random_y_m > 0.0:
            x_m += random.uniform(-random_x_m, random_x_m)
            y_m += random.uniform(-random_y_m, random_y_m)
        elif self.cfg.gazebo_random_xy_radius_m > 0.0:
            angle = random.uniform(0.0, 2.0 * math.pi)
            radius = self.cfg.gazebo_random_xy_radius_m * math.sqrt(random.random())
            x_m += radius * math.cos(angle)
            y_m += radius * math.sin(angle)
        if self.cfg.gazebo_random_z_m > 0.0:
            z_m += random.uniform(-self.cfg.gazebo_random_z_m, self.cfg.gazebo_random_z_m)
        if self.cfg.gazebo_random_yaw_deg > 0.0:
            yaw_rad += math.radians(random.uniform(-self.cfg.gazebo_random_yaw_deg, self.cfg.gazebo_random_yaw_deg))
        return x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad

    def _sample_airborne_pose(self) -> tuple[float, float, float, float, float, float]:
        pose_spec = self.cfg.airborne_reset_pose.strip() or self.cfg.gazebo_base_pose.strip()
        if pose_spec:
            x_m, y_m, _z_m, roll_rad, pitch_rad, yaw_rad = parse_gazebo_pose_spec(
                pose_spec,
                degrees=self.cfg.gazebo_base_pose_degrees,
            )
        else:
            x_m = y_m = roll_rad = pitch_rad = yaw_rad = 0.0
        z_m = self.cfg.airborne_reset_alt_m
        xy_radius_m = self.cfg.airborne_reset_xy_radius_m
        if xy_radius_m < 0.0:
            xy_radius_m = self.cfg.gazebo_random_xy_radius_m
        if xy_radius_m > 0.0:
            angle = random.uniform(0.0, 2.0 * math.pi)
            radius = xy_radius_m * math.sqrt(random.random())
            x_m += radius * math.cos(angle)
            y_m += radius * math.sin(angle)
        yaw_random_deg = self.cfg.airborne_reset_yaw_deg
        if yaw_random_deg < 0.0:
            yaw_random_deg = self.cfg.gazebo_random_yaw_deg
        if yaw_random_deg > 0.0:
            yaw_rad += math.radians(random.uniform(-yaw_random_deg, yaw_random_deg))
        return x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad


class ArduPilotSpiralEnv:
    """Direct online environment backed by ArduPilot SITL."""

    def __init__(self, config: SpiralEnvConfig | None = None):
        self.cfg = config or config_from_env()
        self.process_group = SimulatorProcessGroup(self.cfg)
        if self.cfg.action_space not in {"heading", "heading_altitude"}:
            raise ValueError("action_space must be one of: heading, heading_altitude")
        if self.cfg.action_mode not in {"absolute_k", "delta_k"}:
            raise ValueError("action_mode must be one of: absolute_k, delta_k")
        if self.cfg.command_mode not in {"full_guided", "heading_alt", "weak_goto"}:
            raise ValueError("command_mode must be one of: full_guided, heading_alt, weak_goto")
        self.action_dim = 2 if self.cfg.action_space == "heading_altitude" else 1
        self.observation_dim = 17 if self.action_dim == 2 else 15
        self.action_low = np.full(self.action_dim, -1.0, dtype=np.float32)
        self.action_high = np.full(self.action_dim, 1.0, dtype=np.float32)
        self.master = None
        self.last_gp = None
        self.last_att = None
        self.last_nav = None
        self.center_lat = None
        self.center_lon = None
        self.h_start = None
        self.phi_prev = None
        self.phi_accum = 0.0
        self.turns_completed = 0.0
        self.prev_progress = 0.0
        self.prev_k_heading = (self.cfg.k_min + self.cfg.k_max) / 2.0
        self.prev_k_altitude = (self.cfg.k_alt_min + self.cfg.k_alt_max) / 2.0
        self.current_k_heading = self.prev_k_heading
        self.current_k_altitude = self.prev_k_altitude
        self.wind_enabled = False
        self.wind_enable_time = None
        self.wind_speed_now = 0.0
        self.step_count = 0
        self.episode_reset_count = 0
        self.last_step_time = 0.0
        self.last_position_time_s = None
        self.last_action_values = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self) -> tuple[np.ndarray, dict]:
        first_auto_launch = self.cfg.auto_launch and self.master is None and self.episode_reset_count == 0
        use_airborne_reset = (
            self.cfg.auto_launch
            and self.cfg.airborne_reset_after_first_episode
            and self.episode_reset_count > 0
            and self.master is not None
        )
        if first_auto_launch:
            self._start_initial_simulation()
            self._prepare_vehicle()
        elif use_airborne_reset:
            self._reset_airborne_episode()
        elif self.cfg.auto_launch and (self.cfg.relaunch_every_episode or self.master is None):
            self._restart_simulation()
            self._prepare_vehicle()
        else:
            self._connect()
            self._prepare_vehicle()
        self.episode_reset_count += 1
        self.step_count = 0
        self.prev_progress = 0.0
        self.prev_k_heading = (self.cfg.k_min + self.cfg.k_max) / 2.0
        self.prev_k_altitude = (self.cfg.k_alt_min + self.cfg.k_alt_max) / 2.0
        self.current_k_heading = self.prev_k_heading
        self.current_k_altitude = self.prev_k_altitude
        self.last_action_values = np.zeros(self.action_dim, dtype=np.float32)
        self.last_step_time = time.time()
        self.last_position_time_s = self._message_time_s(self.last_gp)
        obs, info = self._observe_info(self.current_k_heading, self.current_k_altitude, self.last_action_values)
        return obs, info

    def step(self, action: np.ndarray | list[float] | float) -> tuple[np.ndarray, float, bool, dict]:
        action_values = self._normalize_action(action)
        k_heading, k_altitude = self._action_to_gains(action_values)
        _, current_info = self._observe_info(
            self.current_k_heading,
            self.current_k_altitude,
            self.last_action_values,
            advance_phase=False,
        )
        step_start_sim_time_s = self.last_position_time_s
        delta_deg, alt_corr_m = self._send_spiral_command(k_heading, k_altitude, current_info)
        self._wait_command_period(step_start_sim_time_s)
        self._update_wind()
        self.last_action_values = action_values
        obs, info = self._observe_info(k_heading, k_altitude, self.last_action_values)
        info["cmd_heading_deg"] = current_info["cmd_heading_deg"]
        info["heading_correction_deg"] = current_info["heading_correction_deg"]
        info["cmd_alt_m"] = current_info["cmd_alt_m"]
        info["altitude_correction_m"] = current_info["altitude_correction_m"]
        info["goto_sent"] = current_info["goto_sent"]

        done = False
        if info["progress"] >= 1.0:
            done = True
        if abs(info["dr_m"]) >= self.cfg.terminate_radial_error_m:
            done = True
            info["failure_reason"] = "radial_error_limit"
        if info["rel_alt_m"] < self.cfg.min_safe_alt_m:
            done = True
            info["failure_reason"] = "min_altitude_limit"

        self.step_count += 1
        if self.step_count >= self.cfg.max_episode_steps:
            done = True
            info["failure_reason"] = "max_episode_steps"

        reward = self._reward(info, delta_deg, alt_corr_m, k_heading, k_altitude, done)
        self.prev_k_heading = k_heading
        self.prev_k_altitude = k_altitude
        self.prev_progress = info["progress"]
        self.current_k_heading = k_heading
        self.current_k_altitude = k_altitude
        return obs, reward, done, info

    def sample_random_action(self) -> np.ndarray:
        return np.random.uniform(self.action_low, self.action_high).astype(np.float32)

    def close(self, stop_simulators: bool = True) -> None:
        if self.master is not None:
            try:
                clear_rc_override(self.master)
            except Exception:
                pass
            try:
                self.master.close()
            except Exception:
                pass
            self.master = None
        if self.cfg.auto_launch and stop_simulators:
            self.process_group.stop()

    def _restart_simulation(self) -> None:
        self.close(stop_simulators=False)
        self.process_group.prepare_episode()
        self._connect(force_reconnect=True)

    def _start_initial_simulation(self) -> None:
        self.close(stop_simulators=False)
        self.process_group.start()
        self._connect(force_reconnect=True)

    def _reset_airborne_episode(self) -> None:
        try:
            clear_rc_override(self.master)
        except Exception:
            pass
        self.process_group.prepare_airborne_episode()
        self._request_streams()
        self._prepare_airborne_vehicle()

    def _connect(self, force_reconnect: bool = False) -> None:
        if self.master is not None and not force_reconnect:
            return
        if self.master is not None:
            try:
                self.master.close()
            except Exception:
                pass
        self.master = mavutil.mavlink_connection(self.cfg.master)
        wait_heartbeat(self.master, timeout_s=self.cfg.heartbeat_timeout_s)
        self._request_streams()

    def _request_streams(self) -> None:
        request_message_interval(self.master, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, self.cfg.cmd_hz * 2.0)
        request_message_interval(self.master, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, self.cfg.cmd_hz * 2.0)
        request_message_interval(self.master, mavutil.mavlink.MAVLINK_MSG_ID_NAV_CONTROLLER_OUTPUT, self.cfg.cmd_hz * 2.0)

    def _prepare_vehicle(self) -> None:
        self._reset_wind()
        if not self.cfg.skip_takeoff:
            self._takeoff_sequence()
        self._wait_for_spiral_start()
        set_mode(self.master, self.cfg.climb_mode)
        wait_mode(self.master, self.cfg.climb_mode, timeout_s=20.0)
        send_guided_change_speed(self.master, self.cfg.speed)

    def _prepare_airborne_vehicle(self) -> None:
        self._reset_wind()
        self._wait_airframe_ready()
        self._poll_messages(until_position=True)
        rel_alt = self.last_gp.relative_alt / 1000.0
        if rel_alt < self.cfg.min_safe_alt_m:
            raise RuntimeError(
                f"Airborne reset produced rel_alt={rel_alt:.1f} m, below min_safe_alt_m={self.cfg.min_safe_alt_m:.1f} m"
            )
        self._init_spiral_state(rel_alt)
        set_mode(self.master, self.cfg.climb_mode)
        wait_mode(self.master, self.cfg.climb_mode, timeout_s=20.0)
        send_guided_change_speed(self.master, self.cfg.speed)

    def _reset_wind(self) -> None:
        set_param(self.master, "SIM_WIND_SPD", 0.0)
        set_param(self.master, "SIM_WIND_TURB", 0.0)
        self.wind_enabled = False
        self.wind_enable_time = None
        self.wind_speed_now = 0.0

    def _takeoff_sequence(self) -> None:
        if self.cfg.takeoff_mode.upper() == "TAKEOFF":
            self._takeoff_via_takeoff_mode()
            return
        set_mode(self.master, self.cfg.takeoff_mode)
        wait_mode(self.master, self.cfg.takeoff_mode, timeout_s=20.0)
        self._wait_airframe_ready()
        self._arm_with_retries()
        rc_override(self.master, {self.cfg.takeoff_throttle_channel: self.cfg.takeoff_throttle_pwm})

    def _takeoff_via_takeoff_mode(self) -> None:
        # ArduPlane TAKEOFF mode climbs to TKOFF_ALT and then loiters until we switch to GUIDED.
        set_param(self.master, "TKOFF_ALT", self.cfg.spiral_start_alt)
        set_param(self.master, "TKOFF_LVL_ALT", min(self.cfg.takeoff_level_alt_m, self.cfg.spiral_start_alt))
        set_param(self.master, "TKOFF_DIST", self.cfg.takeoff_loiter_dist_m)
        set_param(self.master, "TKOFF_THR_MAX", self.cfg.takeoff_thr_max_pct)
        set_param(self.master, "TKOFF_THR_MINACC", self.cfg.takeoff_thr_minacc)
        set_param(self.master, "TKOFF_THR_MINSPD", self.cfg.takeoff_thr_minspd)
        set_param(self.master, "TKOFF_THR_DELAY", round(self.cfg.takeoff_thr_delay_s * 10.0))
        self._wait_airframe_ready()
        self._arm_with_retries()
        set_mode(self.master, "TAKEOFF")
        wait_mode(self.master, "TAKEOFF", timeout_s=20.0)

    def _wait_airframe_ready(self) -> None:
        deadline = time.time() + max(0.0, self.cfg.prearm_settle_s)
        saw_position = self.last_gp is not None
        saw_attitude = self.last_att is not None
        while time.time() < deadline:
            msg = self.master.recv_match(
                type=["GLOBAL_POSITION_INT", "ATTITUDE", "STATUSTEXT"],
                blocking=True,
                timeout=0.5,
            )
            if msg is None:
                continue
            msg_type = msg.get_type()
            if msg_type == "GLOBAL_POSITION_INT":
                self.last_gp = msg
                saw_position = True
            elif msg_type == "ATTITUDE":
                self.last_att = msg
                saw_attitude = True
            elif msg_type == "STATUSTEXT":
                print(f"AP: {msg.text}")
        if not saw_position:
            self._poll_messages(until_position=True)
        if not saw_attitude:
            self._poll_messages(until_position=False)

    def _arm_with_retries(self) -> None:
        deadline = time.time() + self.cfg.arm_wait_s
        next_arm_time = 0.0
        last_status = ""
        while time.time() < deadline:
            if self.master.motors_armed():
                return
            now = time.time()
            if now >= next_arm_time:
                arm_vehicle(self.master)
                next_arm_time = now + max(0.5, self.cfg.arm_retry_interval_s)
            msg = self.master.recv_match(
                type=["HEARTBEAT", "COMMAND_ACK", "STATUSTEXT"],
                blocking=True,
                timeout=0.5,
            )
            if msg is None:
                continue
            msg_type = msg.get_type()
            if msg_type == "STATUSTEXT":
                last_status = msg.text
                print(f"AP: {msg.text}")
            elif msg_type == "COMMAND_ACK" and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                result = mavutil.mavlink.enums["MAV_RESULT"].get(msg.result)
                result_name = result.name if result is not None else str(msg.result)
                print(f"Got COMMAND_ACK: COMPONENT_ARM_DISARM: {result_name}")
        detail = f": {last_status}" if last_status else ""
        raise TimeoutError(f"Vehicle did not arm within {self.cfg.arm_wait_s} s{detail}")

    def _wait_for_spiral_start(self) -> None:
        deadline = time.time() + self.cfg.ready_timeout_s
        next_throttle_refresh = 0.0
        needs_manual_takeoff_throttle = (not self.cfg.skip_takeoff) and self.cfg.takeoff_mode.upper() != "TAKEOFF"
        while time.time() < deadline:
            now = time.time()
            if needs_manual_takeoff_throttle and now >= next_throttle_refresh:
                rc_override(self.master, {self.cfg.takeoff_throttle_channel: self.cfg.takeoff_throttle_pwm})
                next_throttle_refresh = now + max(0.05, self.cfg.takeoff_throttle_refresh_s)
            self._poll_messages(until_position=True)
            rel_alt = self.last_gp.relative_alt / 1000.0
            if rel_alt >= self.cfg.spiral_start_alt:
                rc_override(self.master, {self.cfg.takeoff_throttle_channel: self.cfg.takeoff_release_pwm})
                self._init_spiral_state(rel_alt)
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"UAV did not reach spiral_start_alt={self.cfg.spiral_start_alt} m "
            f"within {self.cfg.ready_timeout_s} s"
        )

    def _poll_messages(self, until_position: bool = False) -> None:
        deadline = time.time() + self.cfg.step_wall_timeout_s
        got_position = False
        while time.time() < deadline:
            msg = self.master.recv_match(
                type=["GLOBAL_POSITION_INT", "ATTITUDE", "NAV_CONTROLLER_OUTPUT"],
                blocking=until_position and not got_position,
                timeout=0.1,
            )
            if msg is None:
                if until_position and not got_position:
                    continue
                break
            msg_type = msg.get_type()
            if msg_type == "GLOBAL_POSITION_INT":
                self.last_gp = msg
                self.last_position_time_s = self._message_time_s(msg)
                got_position = True
            elif msg_type == "ATTITUDE":
                self.last_att = msg
            elif msg_type == "NAV_CONTROLLER_OUTPUT":
                self.last_nav = msg
            if until_position and got_position:
                break
        if until_position and not got_position:
            raise TimeoutError("No GLOBAL_POSITION_INT message received from ArduPilot")

    def _message_time_s(self, msg) -> float | None:
        if msg is None:
            return None
        time_boot_ms = getattr(msg, "time_boot_ms", None)
        if time_boot_ms is None:
            return None
        return float(time_boot_ms) / 1000.0

    def _poll_until_sim_time(self, target_time_s: float) -> None:
        deadline = time.time() + self.cfg.step_wall_timeout_s
        got_position = False
        while time.time() < deadline:
            msg = self.master.recv_match(
                type=["GLOBAL_POSITION_INT", "ATTITUDE", "NAV_CONTROLLER_OUTPUT"],
                blocking=True,
                timeout=0.1,
            )
            if msg is None:
                continue
            msg_type = msg.get_type()
            if msg_type == "GLOBAL_POSITION_INT":
                self.last_gp = msg
                self.last_position_time_s = self._message_time_s(msg)
                got_position = True
                if self.last_position_time_s is not None and self.last_position_time_s >= target_time_s:
                    return
            elif msg_type == "ATTITUDE":
                self.last_att = msg
            elif msg_type == "NAV_CONTROLLER_OUTPUT":
                self.last_nav = msg
        if got_position:
            last_time = self.last_position_time_s
            raise TimeoutError(
                f"Simulation time did not reach {target_time_s:.3f} s within "
                f"{self.cfg.step_wall_timeout_s:.1f} wall seconds; last GLOBAL_POSITION_INT time was "
                f"{last_time:.3f} s"
            )
        raise TimeoutError("No GLOBAL_POSITION_INT message received from ArduPilot")

    def _init_spiral_state(self, rel_alt: float) -> None:
        lat = self.last_gp.lat / 1e7
        lon = self.last_gp.lon / 1e7
        radial = math.radians(self.cfg.start_radial_deg)
        north_vec = self.cfg.r0 * math.cos(radial)
        east_vec = self.cfg.r0 * math.sin(radial)
        self.center_lat, self.center_lon = shift_latlon(lat, lon, -north_vec, -east_vec)
        n_now, e_now = latlon_to_ne(lat, lon, self.center_lat, self.center_lon)
        self.h_start = rel_alt
        self.phi_prev = math.atan2(e_now, n_now)
        self.phi_accum = 0.0
        self.turns_completed = 0.0

    def _update_wind(self) -> None:
        rel_alt = self.last_gp.relative_alt / 1000.0
        if (not self.wind_enabled) and rel_alt >= self.cfg.wind_enable_alt and self.cfg.wind_speed_mps > 0.0:
            self.wind_enabled = True
            self.wind_enable_time = self.last_position_time_s if self.last_position_time_s is not None else time.time()
            set_param(self.master, "SIM_WIND_TURB", self.cfg.wind_turb)
            set_param(self.master, "SIM_WIND_DIR", self.cfg.wind_from_deg)

        if not self.wind_enabled:
            return
        if self.cfg.wind_ramp_s <= 0.0:
            self.wind_speed_now = self.cfg.wind_speed_mps
        else:
            now_s = self.last_position_time_s if self.last_position_time_s is not None else time.time()
            elapsed = max(0.0, now_s - self.wind_enable_time)
            self.wind_speed_now = clamp(
                self.cfg.wind_speed_mps * elapsed / self.cfg.wind_ramp_s,
                0.0,
                self.cfg.wind_speed_mps,
            )
        set_param(self.master, "SIM_WIND_SPD", self.wind_speed_now)

    def _wait_command_period(self, step_start_sim_time_s: float | None = None) -> None:
        cmd_dt = 1.0 / max(1e-6, self.cfg.cmd_hz)
        if self.cfg.use_sim_time_control and step_start_sim_time_s is not None:
            self._poll_until_sim_time(step_start_sim_time_s + cmd_dt)
            self.last_step_time = time.time()
            return
        elapsed = time.time() - self.last_step_time
        if elapsed < cmd_dt:
            time.sleep(cmd_dt - elapsed)
        self.last_step_time = time.time()
        self._poll_messages(until_position=True)

    def _normalize_action(self, action: np.ndarray | list[float] | float) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size == 0:
            values = np.zeros(self.action_dim, dtype=np.float32)
        if values.size < self.action_dim:
            values = np.pad(values, (0, self.action_dim - values.size))
        values = values[: self.action_dim]
        return np.clip(values, self.action_low, self.action_high).astype(np.float32)

    def _action_to_gain(self, action_value: float) -> float:
        if self.cfg.action_mode == "delta_k":
            span = self.cfg.k_max - self.cfg.k_min
            k = self.current_k_heading + 0.1 * span * clamp(action_value, -1.0, 1.0)
        else:
            alpha = 0.5 * (clamp(action_value, -1.0, 1.0) + 1.0)
            k = self.cfg.k_min + alpha * (self.cfg.k_max - self.cfg.k_min)
        return clamp(k, self.cfg.k_min, self.cfg.k_max)

    def _action_to_altitude_gain(self, action_value: float) -> float:
        if self.cfg.action_mode == "delta_k":
            span = self.cfg.k_alt_max - self.cfg.k_alt_min
            k = self.current_k_altitude + 0.1 * span * clamp(action_value, -1.0, 1.0)
        else:
            alpha = 0.5 * (clamp(action_value, -1.0, 1.0) + 1.0)
            k = self.cfg.k_alt_min + alpha * (self.cfg.k_alt_max - self.cfg.k_alt_min)
        return clamp(k, self.cfg.k_alt_min, self.cfg.k_alt_max)

    def _action_to_gains(self, action_values: np.ndarray) -> tuple[float, float]:
        k_heading = self._action_to_gain(float(action_values[0]))
        if self.action_dim >= 2:
            k_altitude = self._action_to_altitude_gain(float(action_values[1]))
        else:
            k_altitude = self.current_k_altitude
        return k_heading, k_altitude

    def _normalize_k_heading(self, k_heading: float) -> float:
        k_mid = (self.cfg.k_min + self.cfg.k_max) / 2.0
        k_half = max(1e-6, (self.cfg.k_max - self.cfg.k_min) / 2.0)
        return clamp((k_heading - k_mid) / k_half, -1.0, 1.0)

    def _normalize_k_altitude(self, k_altitude: float) -> float:
        k_mid = (self.cfg.k_alt_min + self.cfg.k_alt_max) / 2.0
        k_half = max(1e-6, (self.cfg.k_alt_max - self.cfg.k_alt_min) / 2.0)
        return clamp((k_altitude - k_mid) / k_half, -1.0, 1.0)

    def _observe_info(
        self,
        k_heading: float,
        k_altitude: float,
        action_values: np.ndarray,
        advance_phase: bool = True,
    ) -> tuple[np.ndarray, dict]:
        lat = self.last_gp.lat / 1e7
        lon = self.last_gp.lon / 1e7
        rel_alt = self.last_gp.relative_alt / 1000.0
        n_now, e_now = latlon_to_ne(lat, lon, self.center_lat, self.center_lon)
        r_now = math.hypot(n_now, e_now)
        phi = math.atan2(e_now, n_now)
        if advance_phase:
            dphi = wrap_pi(phi - self.phi_prev)
            self.phi_prev = phi
            self.phi_accum += dphi
            self.turns_completed = clamp(abs(self.phi_accum) / (2.0 * math.pi), 0.0, self.cfg.turns)

        progress = self.turns_completed / max(1e-6, self.cfg.turns)
        r_ref = self.cfg.r0 + self.cfg.r_step * self.turns_completed
        h_ref = self.h_start + self.cfg.spiral_climb * progress
        dr = r_now - r_ref
        alt_error = rel_alt - h_ref
        dir_sign = -1.0 if self.cfg.direction.upper() == "CW" else 1.0
        start_phi = math.radians(self.cfg.start_radial_deg)
        phi_ref = start_phi + dir_sign * 2.0 * math.pi * self.turns_completed
        ref_n = r_ref * math.cos(phi_ref)
        ref_e = r_ref * math.sin(phi_ref)
        ref_lat, ref_lon = ne_to_latlon(ref_n, ref_e, self.center_lat, self.center_lon)

        lookahead_turns = max(0.0, self.cfg.lookahead_deg) / 360.0
        target_turns_completed = clamp(
            self.turns_completed + lookahead_turns,
            0.0,
            self.cfg.turns,
        )
        target_progress = target_turns_completed / max(1e-6, self.cfg.turns)
        r_tgt = self.cfg.r0 + self.cfg.r_step * target_turns_completed
        h_tgt = self.h_start + self.cfg.spiral_climb * target_progress
        phi_tgt = start_phi + dir_sign * 2.0 * math.pi * target_turns_completed
        tgt_n = r_tgt * math.cos(phi_tgt)
        tgt_e = r_tgt * math.sin(phi_tgt)
        tgt_lat, tgt_lon = ne_to_latlon(tgt_n, tgt_e, self.center_lat, self.center_lon)
        dphi_dt = dir_sign * 2.0 * math.pi
        dn_dt = self.cfg.r_step * math.cos(phi_tgt) - r_tgt * math.sin(phi_tgt) * dphi_dt
        de_dt = self.cfg.r_step * math.sin(phi_tgt) + r_tgt * math.cos(phi_tgt) * dphi_dt
        tgt_heading = wrap_360(math.degrees(math.atan2(de_dt, dn_dt)))
        n_err, e_err = ne_error_from_latlon(lat, lon, tgt_lat, tgt_lon)
        e_ct = cross_track_error(n_err, e_err, tgt_heading)

        roll_deg = pitch_deg = yaw_deg = 0.0
        if self.last_att is not None:
            roll_deg = math.degrees(self.last_att.roll)
            pitch_deg = math.degrees(self.last_att.pitch)
            yaw_deg = wrap_360(math.degrees(self.last_att.yaw))
        nav_roll_deg = nav_pitch_deg = nav_bearing_deg = target_bearing_deg = 0.0
        nav_alt_error_m = nav_aspd_error_mps = nav_xtrack_error_m = 0.0
        if self.last_nav is not None:
            nav_roll_deg = float(getattr(self.last_nav, "nav_roll", 0.0))
            nav_pitch_deg = float(getattr(self.last_nav, "nav_pitch", 0.0))
            nav_bearing_deg = float(getattr(self.last_nav, "nav_bearing", 0.0))
            target_bearing_deg = float(getattr(self.last_nav, "target_bearing", 0.0))
            nav_alt_error_m = float(getattr(self.last_nav, "alt_error", 0.0))
            nav_aspd_error_mps = float(getattr(self.last_nav, "aspd_error", 0.0))
            nav_xtrack_error_m = float(getattr(self.last_nav, "xtrack_error", 0.0))

        action_heading = float(action_values[0]) if len(action_values) >= 1 else 0.0
        action_altitude = float(action_values[1]) if len(action_values) >= 2 else 0.0
        obs_values = [
            clamp(dr / 25.0, -10.0, 10.0),
            clamp(e_ct / 25.0, -10.0, 10.0),
            clamp(alt_error / 20.0, -10.0, 10.0),
            math.sin(phi),
            math.cos(phi),
            progress,
            self._normalize_k_heading(k_heading),
        ]
        if self.action_dim >= 2:
            obs_values.append(self._normalize_k_altitude(k_altitude))
        obs_values.extend(
            [
                clamp(self.wind_speed_now / 15.0, 0.0, 10.0),
                math.sin(math.radians(self.cfg.wind_from_deg)),
                math.cos(math.radians(self.cfg.wind_from_deg)),
                clamp(roll_deg / 45.0, -10.0, 10.0),
                clamp(pitch_deg / 30.0, -10.0, 10.0),
                math.sin(math.radians(yaw_deg)),
                math.cos(math.radians(yaw_deg)),
                clamp(action_heading, -1.0, 1.0),
            ]
        )
        if self.action_dim >= 2:
            obs_values.append(clamp(action_altitude, -1.0, 1.0))
        obs = np.array(obs_values, dtype=np.float32)
        info = {
            "lat_deg": lat,
            "lon_deg": lon,
            "rel_alt_m": rel_alt,
            "center_lat_deg": self.center_lat,
            "center_lon_deg": self.center_lon,
            "reference_lat_deg": ref_lat,
            "reference_lon_deg": ref_lon,
            "reference_alt_m": h_ref,
            "r_ref_m": r_ref,
            "r_now_m": r_now,
            "dr_m": dr,
            "abs_dr_m": abs(dr),
            "alt_error_m": alt_error,
            "progress": progress,
            "turns_completed": self.turns_completed,
            "target_lat_deg": tgt_lat,
            "target_lon_deg": tgt_lon,
            "target_alt_m": h_tgt,
            "target_progress": target_progress,
            "target_turns_completed": target_turns_completed,
            "target_heading_deg": tgt_heading,
            "cross_track_error_m": e_ct,
            "K": k_heading,
            "K_heading": k_heading,
            "K_altitude": k_altitude,
            "action_heading": action_heading,
            "action_altitude": action_altitude if self.action_dim >= 2 else 0.0,
            "command_mode": self.cfg.command_mode,
            "wind_speed_mps": self.wind_speed_now,
            "time_boot_s": self.last_position_time_s if self.last_position_time_s is not None else 0.0,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "nav_roll_deg": nav_roll_deg,
            "nav_pitch_deg": nav_pitch_deg,
            "nav_bearing_deg": nav_bearing_deg,
            "target_bearing_deg": target_bearing_deg,
            "nav_alt_error_m": nav_alt_error_m,
            "nav_aspd_error_mps": nav_aspd_error_mps,
            "nav_xtrack_error_m": nav_xtrack_error_m,
        }
        return obs, info

    def _should_send_goto(self) -> bool:
        if self.cfg.command_mode == "heading_alt":
            return False
        if self.cfg.command_mode == "full_guided":
            return True
        interval = max(1, int(self.cfg.goto_every_steps))
        return self.step_count % interval == 0

    def _send_spiral_command(self, k_heading: float, k_altitude: float, info: dict) -> tuple[float, float]:
        delta_deg = math.degrees(math.atan2(info["cross_track_error_m"], max(1e-3, self.cfg.l_corr))) * k_heading
        delta_deg = clamp(delta_deg, -self.cfg.max_corr_deg, self.cfg.max_corr_deg)
        cmd_heading = wrap_360(info["target_heading_deg"] + delta_deg)
        altitude_correction_m = 0.0
        if self.action_dim >= 2:
            altitude_correction_m = clamp(
                -k_altitude * info["alt_error_m"],
                -self.cfg.max_alt_corr_m,
                self.cfg.max_alt_corr_m,
            )
        cmd_alt_m = max(self.cfg.min_safe_alt_m, info["target_alt_m"] + altitude_correction_m)
        goto_sent = self._should_send_goto()
        if goto_sent:
            send_guided_goto(self.master, info["target_lat_deg"], info["target_lon_deg"], cmd_alt_m)
        send_guided_change_heading(self.master, cmd_heading, max_centrip_acc=self.cfg.centrip_acc)
        if self.cfg.command_mode == "heading_alt":
            send_guided_change_alt(self.master, cmd_alt_m, rate_mps=self.cfg.climb_rate)
        info["cmd_heading_deg"] = cmd_heading
        info["heading_correction_deg"] = delta_deg
        info["cmd_alt_m"] = cmd_alt_m
        info["altitude_correction_m"] = altitude_correction_m
        info["goto_sent"] = int(goto_sent)
        return delta_deg, altitude_correction_m

    def _reward(
        self,
        info: dict,
        delta_deg: float,
        alt_corr_m: float,
        k_heading: float,
        k_altitude: float,
        done: bool,
    ) -> float:
        radial_error_scale = max(1e-6, self.cfg.expected_radial_error_m)
        altitude_error_scale = max(1e-6, self.cfg.expected_alt_error_m)
        heading_scale = max(1e-6, self.cfg.max_corr_deg)
        k_scale = max(1e-6, self.cfg.k_max - self.cfg.k_min)
        k_alt_scale = max(1e-6, self.cfg.k_alt_max - self.cfg.k_alt_min)
        progress_delta = max(0.0, info["progress"] - self.prev_progress)
        heading_smooth = abs(k_heading - self.prev_k_heading) / k_scale
        altitude_smooth = 0.0
        if self.action_dim >= 2:
            altitude_smooth = abs(k_altitude - self.prev_k_altitude) / k_alt_scale
        smooth_abs = 0.5 * (heading_smooth + altitude_smooth) if self.action_dim >= 2 else heading_smooth

        if self.cfg.reward_mode == "band_bonus":
            radial_term = (info["abs_dr_m"] / radial_error_scale) ** 2
            alt_term = (abs(info["alt_error_m"]) / altitude_error_scale) ** 2
            heading_term = (abs(delta_deg) / heading_scale) ** 2
            altitude_correction_term = (abs(alt_corr_m) / max(1e-6, self.cfg.max_alt_corr_m)) ** 2
            smooth_term = smooth_abs**2
            reward = (
                self.cfg.reward_progress_weight * progress_delta
                - self.cfg.reward_radial_weight * radial_term
                - self.cfg.reward_alt_weight * alt_term
                - self.cfg.reward_heading_weight * 0.5 * (heading_term + altitude_correction_term)
                - self.cfg.reward_smooth_weight * smooth_term
            )
            if (
                info["abs_dr_m"] <= self.cfg.expected_radial_error_m
                and abs(info["alt_error_m"]) <= self.cfg.expected_alt_error_m
            ):
                reward += self.cfg.reward_in_band_bonus
        elif self.cfg.reward_mode == "continuous_error":
            radial_term = info["abs_dr_m"] / radial_error_scale
            alt_term = abs(info["alt_error_m"]) / altitude_error_scale
            cross_track_term = abs(info["cross_track_error_m"]) / radial_error_scale
            heading_term = abs(delta_deg) / heading_scale
            altitude_correction_term = abs(alt_corr_m) / max(1e-6, self.cfg.max_alt_corr_m)
            smooth_term = smooth_abs
            reward = (
                self.cfg.reward_progress_weight * progress_delta
                - self.cfg.reward_radial_weight * radial_term
                - self.cfg.reward_alt_weight * alt_term
                - self.cfg.reward_cross_track_weight * cross_track_term
                - self.cfg.reward_heading_weight * 0.5 * (heading_term + altitude_correction_term)
                - self.cfg.reward_smooth_weight * smooth_term
                - self.cfg.reward_step_penalty
            )
        else:
            raise ValueError("reward_mode must be one of: continuous_error, band_bonus")

        if done and info["progress"] >= 1.0:
            reward += self.cfg.reward_completion_bonus
        if done and info["progress"] < 1.0:
            reward -= self.cfg.reward_failure_penalty
        return float(reward)
