"""ABS strike-zone geometry.

Verified 2026-10-21 against Baseball Savant's embedded `vizChallenges` rows
(17 regular-season challenges, four games): `midplate_location` reproduces
Savant's `plateX/plateZ` and `edge_distance_in` reproduces `edge_dist_calc`
to 1e-5. The fixture is in tests/savant_vizchallenges_2026-09-21_to_27.json.

Conventions (feet unless noted; MLB StatsAPI `pitchData.coordinates`):
  y0 = 50 ft reference plane; y decreases toward the catcher.
  Front edge of home plate is at y = 17/12 ft; ABS judges the ball at the
  plate midpoint, 8.5 in behind the front edge.
  Zone is 17 in wide (+/- 8.5 in). Top/bottom come from the feed's
  strikeZoneTop/strikeZoneBottom, which in 2026 are the fixed ABS bounds
  (53.5% and 27% of measured height; top/bottom ratio is 1.981 on every
  batter checked). A pitch is an ABS strike if ANY part of the ball
  (diameter 2.9 in) overlaps the zone at the midpoint plane.
"""
from __future__ import annotations

import math

PLATE_FRONT_Y = 17 / 12          # ft
PLATE_MID_Y = PLATE_FRONT_Y - 8.5 / 12
ZONE_HALF_WIDTH = 8.5 / 12       # ft
BALL_RADIUS_IN = 2.9 / 2
ABS_TOP_FRAC = 0.535
ABS_BOT_FRAC = 0.27


def solve_at_y(c: dict, y_target: float) -> tuple[float, float, float]:
    """Return (x, z, t) where the 9-parameter trajectory crosses y_target."""
    a = 0.5 * c["aY"]
    b = c["vY0"]
    cc = c["y0"] - y_target
    t = (-b - math.sqrt(b * b - 4 * a * cc)) / (2 * a)
    x = c["x0"] + c["vX0"] * t + 0.5 * c["aX"] * t * t
    z = c["z0"] + c["vZ0"] * t + 0.5 * c["aZ"] * t * t
    return x, z, t


def midplate_location(c: dict) -> tuple[float, float]:
    """Ball-center (x, z) at the plate midpoint, the plane ABS judges."""
    x, z, _ = solve_at_y(c, PLATE_MID_Y)
    return x, z


def edge_distance_in(x: float, z: float, top: float, bot: float) -> float:
    """Signed inches from the ball's EDGE to the nearest zone boundary.

    Negative means the ball overlaps the zone (ABS strike); positive means
    it misses (ABS ball). Matches Savant's `edge_dist_calc` exactly.
    """
    dx = abs(x) - ZONE_HALF_WIDTH
    dz = max(bot - z, z - top)
    if dx <= 0 and dz <= 0:
        center_ft = -min(-dx, -dz)
    else:
        center_ft = math.hypot(max(dx, 0.0), max(dz, 0.0))
    return center_ft * 12 - BALL_RADIUS_IN


def abs_is_strike(x: float, z: float, top: float, bot: float) -> bool:
    return edge_distance_in(x, z, top, bot) <= 0


def implied_height_in(top: float, bot: float) -> float:
    """Measured height implied by the ABS bounds (average of the two)."""
    return 12 * (top / ABS_TOP_FRAC + bot / ABS_BOT_FRAC) / 2


# --------------------------------------------------------------------------- #
# Vectorised versions for the pitch table (numpy / pandas columns)
# --------------------------------------------------------------------------- #
def midplate_location_np(x0, y0, z0, vx0, vy0, vz0, ax, ay, az, y_target: float = PLATE_MID_Y):
    import numpy as np

    disc = vy0 * vy0 - 2.0 * ay * (y0 - y_target)
    t = (-vy0 - np.sqrt(disc)) / ay
    x = x0 + vx0 * t + 0.5 * ax * t * t
    z = z0 + vz0 * t + 0.5 * az * t * t
    return x, z


def edge_distance_in_np(x, z, top, bot):
    import numpy as np

    dx = np.abs(x) - ZONE_HALF_WIDTH
    dz = np.maximum(bot - z, z - top)
    inside = (dx <= 0) & (dz <= 0)
    center = np.where(inside, np.maximum(dx, dz), np.hypot(np.maximum(dx, 0.0), np.maximum(dz, 0.0)))
    return center * 12 - BALL_RADIUS_IN
