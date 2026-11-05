"""
Lane Departure Warning (LDW) Module.

Uses lane_center_offset (already computed by features.py) to trigger
a visual and logged alert when the vehicle drifts toward a lane boundary.
"""

from dataclasses import dataclass
from enum import Enum


class DepartureStatus(Enum):
    SAFE = "SAFE"
    WARNING_LEFT = "WARNING LEFT"
    WARNING_RIGHT = "WARNING RIGHT"
    CRITICAL_LEFT = "CRITICAL LEFT — DEPART"
    CRITICAL_RIGHT = "CRITICAL RIGHT — DEPART"


@dataclass
class LDWResult:
    status: DepartureStatus
    offset_px: float          # Signed pixel offset from lane center (- = left, + = right)
    offset_meters: float      # Same in real-world meters
    alert: bool               # True if driver intervention is recommended


# Physical calibration constants
# Derived from TuSimple: lane width ≈ 3.7m, image width = 1280px
LANE_WIDTH_PX = 640.0           # Half-lane in pixels (image center = lane center assumption)
LANE_WIDTH_METERS = 3.7         # Standard US/global lane width
PIXELS_PER_METER = LANE_WIDTH_PX / (LANE_WIDTH_METERS / 2.0)   # px/m

# Thresholds (tune these)
WARNING_THRESHOLD_M = 0.4       # 40cm from center → warning
CRITICAL_THRESHOLD_M = 0.8      # 80cm from center → critical (nearly at marking)


def compute_ldw(lane_center_offset_px: float,
                direction_prediction: str) -> LDWResult:
    """
    Compute lane departure status from lane_center_offset.

    Args:
        lane_center_offset_px: Output of features.py — signed horizontal
                               offset of detected lane center from image
                               center column (640px). Negative = left drift.
        direction_prediction:  Current classifier verdict ("left"/"straight"/"right")

    Returns:
        LDWResult with status, offset in meters, and alert flag.
    """
    offset_m = lane_center_offset_px / PIXELS_PER_METER
    abs_offset = abs(offset_m)

    # Only warn on straight roads — turning intentionally is not departure
    if direction_prediction != "straight":
        return LDWResult(
            status=DepartureStatus.SAFE,
            offset_px=lane_center_offset_px,
            offset_meters=offset_m,
            alert=False
        )

    if abs_offset < WARNING_THRESHOLD_M:
        status = DepartureStatus.SAFE
        alert = False

    elif abs_offset < CRITICAL_THRESHOLD_M:
        status = DepartureStatus.WARNING_LEFT if offset_m < 0 else DepartureStatus.WARNING_RIGHT
        alert = True

    else:
        status = DepartureStatus.CRITICAL_LEFT if offset_m < 0 else DepartureStatus.CRITICAL_RIGHT
        alert = True

    return LDWResult(
        status=status,
        offset_px=lane_center_offset_px,
        offset_meters=offset_m,
        alert=alert
    )

