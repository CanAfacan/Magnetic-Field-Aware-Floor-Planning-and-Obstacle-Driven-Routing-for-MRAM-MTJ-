# wiring.py
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Union

Point = Tuple[float, float]
CircleSpec = Tuple[float, float, float]
RectSpec = Tuple[float, float, float, float]  # x, y, w, h
ObstacleDict = Dict[str, Union[List[CircleSpec], List[RectSpec]]]


CLEARANCE   = 6.0          # extra gap between wires and obstacles (floor-units)
MAX_EXPAND  = 2000.0       # searches envelope for detours
MAX_STEPS   = 500          # iteration cap for Manhattan search


# 
#  Primitive geometry 
# 

def _intersects(p: Point, q: Point, circ: CircleSpec) -> bool:
    """ReturnsTrue if line segment (p to q) intersects/touches the circle."""
    cx, cy, r0 = circ
    r = r0 + CLEARANCE
    (x1, y1), (x2, y2) = p, q

    dx, dy = x2 - x1, y2 - y1
    seg2   = dx * dx + dy * dy
    if seg2 == 0:
        return (x1 - cx) ** 2 + (y1 - cy) ** 2 < r * r

    # closest-approach parameter along the segment
    t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / seg2))
    px, py = x1 + t * dx, y1 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 < r * r


def _rect_intersects(p: Point, q: Point, rect: RectSpec) -> bool:
    """Robustly detect if segment (p to q) touches/enters axis-aligned rectangle.
    The rectangle is first inflated by CLEARANCE on all sides, so a path that
    even *touches* the block edge is rejected.  Uses the Liang-Barsky clip test.
    """
    # Inflates rectangle by CLEARANCE.
    x_min, y_min = rect[0] - CLEARANCE, rect[1] - CLEARANCE
    x_max, y_max = rect[0] + rect[2] + CLEARANCE, rect[1] + rect[3] + CLEARANCE

    x1, y1 = p
    x2, y2 = q

    # Quick reject if both endpoints are strictly on one side.
    if (x1 <= x_min and x2 <= x_min) or (x1 >= x_max and x2 >= x_max):
        return False
    if (y1 <= y_min and y2 <= y_min) or (y1 >= y_max and y2 >= y_max):
        return False
    # Liang-Barsky parameters p, q
    dx, dy = x2 - x1, y2 - y1
    p_vals = (-dx, dx, -dy, dy)
    q_vals = (x1 - x_min, x_max - x1, y1 - y_min, y_max - y1)

    u_enter, u_leave = 0.0, 1.0
    for p_i, q_i in zip(p_vals, q_vals):
        if abs(p_i) < 1e-12:          # segment parallel to this boundary
            if q_i < 0:               # outside and parallel no hit
                return False
            continue                  # inside & parallel  nothing to do
        t = q_i / p_i
        if p_i < 0:                   # entering boundary
            u_enter = max(u_enter, t)
        else:                         # leaving boundary
            u_leave = min(u_leave, t)
        if u_enter > u_leave:
            return False              # exited before entered  no intersection

    return True                       # segment intersects/touches rectangle


def _clear(p: Point, q: Point, obs: ObstacleDict) -> bool:
    """True if the straight segment p to q avoids all obstacles."""
    return (not any(_intersects(p, q, c) for c in obs.get('circles', [])) and
            not any(_rect_intersects(p, q, r) for r in obs.get('rects', [])))

# 
#  Routing kernels
# 

def manhattan_route(p1: Point, p2: Point, obs: ObstacleDict) -> List[Point]:
    """Return an orthogonal (H/V) path p1…p2 that clears obstacles.
    It first tries the two simple L-shapes, then searches outward for a
    Z-shaped detour.  If no legal path exists an empty list is returned.
    """
    x1, y1 = p1
    x2, y2 = p2

    #1) immediate L-shape trials 
    via1, via2 = (x2, y1), (x1, y2)
    if _clear(p1, via1, obs) and _clear(via1, p2, obs):
        return [p1, via1, p2]
    if _clear(p1, via2, obs) and _clear(via2, p2, obs):
        return [p1, via2, p2]

    #2) expanding Z-shape search 
    offset = CLEARANCE
    steps  = 0
    while offset <= MAX_EXPAND and steps < MAX_STEPS:
        for sign in (-1, 1):
            # Horizontal-first Z
            via_h1 = (x1 + sign * offset, y1)
            via_h2 = (x1 + sign * offset, y2)
            if (_clear(p1, via_h1, obs) and _clear(via_h1, via_h2, obs) and
                    _clear(via_h2, p2, obs)):
                return [p1, via_h1, via_h2, p2]

            # Vertical-first Z
            via_v1 = (x1, y1 + sign * offset)
            via_v2 = (x2, y1 + sign * offset)
            if (_clear(p1, via_v1, obs) and _clear(via_v1, via_v2, obs) and
                    _clear(via_v2, p2, obs)):
                return [p1, via_v1, via_v2, p2]
        offset += CLEARANCE * 1.5
        steps  += 1

    #3) give up 
    return []


def euclidean_route(p1: Point, p2: Point, obs: ObstacleDict) -> List[Point]:
    """Shallow Euclidean (straight-line) router.  Returns direct segment if
    it is clear, otherwise the empty list (no attempt to detour).
    """
    return [p1, p2] if _clear(p1, p2, obs) else []


# Steiner 

def _median(xs: List[float]) -> float:
    xs_sorted = sorted(xs)
    mid = len(xs_sorted) // 2
    if len(xs_sorted) % 2:
        return xs_sorted[mid]
    return 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])


def steiner_route(points: List[Point], obs: ObstacleDict) -> List[Tuple[Point, Point]]:

    n = len(points)
    if n < 2:
        return []
    if n == 2:
        path = manhattan_route(points[0], points[1], obs)
        return list(zip(path, path[1:])) if len(path) > 1 else []

    # 1) connects each pin horizontally to (median_x, y_pin)
    med_x = _median([x for x, _ in points])

    segs: List[Tuple[Point, Point]] = []     # all route segments
    trunk_hits: List[Point] = []             # where each pin meets the trunk

    for p in points:
        target = (med_x, p[1])
        path = manhattan_route(p, target, obs)

        # If the straight shot fails (rare)
        if not path:
            offset = CLEARANCE
            found  = False
            while offset <= MAX_EXPAND and not found:
                for sign in (-1, 1):
                    alt_target = (med_x + sign * offset, p[1])
                    path = manhattan_route(p, alt_target, obs)
                    if path:
                        found = True
                        break
                offset += CLEARANCE
        if not path:
            continue

        segs.extend(zip(path, path[1:]))
        trunk_hits.append(path[-1])

    # If <2 pins survived, we are done.
    if len(trunk_hits) < 2:
        return segs

    #  2) connect trunk hits together top to bottom 
    trunk_hits.sort(key=lambda pt: pt[1])  # by y

    a = trunk_hits[0]
    for b in trunk_hits[1:]:
        path = manhattan_route(a, b, obs)
        if len(path) > 1:
            segs.extend(zip(path, path[1:]))
            a = b  # continues the chain from the last point actually connected

    return segs
