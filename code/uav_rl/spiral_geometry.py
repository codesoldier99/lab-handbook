"""Geometry helpers for conical-spiral UAV mapping experiments."""

import math

M_PER_DEG = 111319.49079327357


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_360(deg: float) -> float:
    deg = deg % 360.0
    return deg if deg >= 0.0 else deg + 360.0


def wrap_pi(rad: float) -> float:
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def latlon_to_ne(lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float) -> tuple[float, float]:
    dlat = lat_deg - lat0_deg
    dlon = lon_deg - lon0_deg
    north = dlat * M_PER_DEG
    east = dlon * M_PER_DEG * math.cos(math.radians(lat0_deg))
    return north, east


def ne_to_latlon(north_m: float, east_m: float, lat0_deg: float, lon0_deg: float) -> tuple[float, float]:
    dlat = north_m / M_PER_DEG
    dlon = east_m / (M_PER_DEG * math.cos(math.radians(lat0_deg)))
    return lat0_deg + dlat, lon0_deg + dlon


def shift_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    return ne_to_latlon(north_m, east_m, lat_deg, lon_deg)


def ne_error_from_latlon(
    lat_deg: float,
    lon_deg: float,
    target_lat_deg: float,
    target_lon_deg: float,
) -> tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    dlat = target_lat_deg - lat_deg
    dlon = target_lon_deg - lon_deg
    north = dlat * M_PER_DEG
    east = dlon * M_PER_DEG * math.cos(lat_rad)
    return north, east


def cross_track_error(n_err: float, e_err: float, target_heading_deg: float) -> float:
    chi = math.radians(wrap_360(target_heading_deg))
    right_n = -math.sin(chi)
    right_e = math.cos(chi)
    return n_err * right_n + e_err * right_e


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def rms(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / max(1, len(xs)))

