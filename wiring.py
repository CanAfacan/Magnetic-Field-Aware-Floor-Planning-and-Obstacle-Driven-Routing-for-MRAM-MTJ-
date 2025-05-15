# wiring.py 
from __future__ import annotations
import math
from typing import List, Tuple

Point      = Tuple[float, float]
CircleSpec = Tuple[float, float, float]

CLEARANCE   = 4.0          # tracks of extra air‑gap
MAX_EXPAND  = 2000.0     
MAX_STEPS   = 500         


def _intersects(p: Point, q: Point, circ: CircleSpec) -> bool:
    cx, cy, r0 = circ
    r = r0 + CLEARANCE
    (x1, y1), (x2, y2) = p, q

    dx, dy = x2 - x1, y2 - y1
    seg2   = dx * dx + dy * dy
    if seg2 == 0:
        return (x1 - cx) ** 2 + (y1 - cy) ** 2 < r * r

    t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / seg2))
    px, py = x1 + t * dx, y1 + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 < r * r


def _clear(p: Point, q: Point, obs: List[CircleSpec]) -> bool:
    return not any(_intersects(p, q, c) for c in obs)

def manhattan_route(p1: Point, p2: Point,
                    obs: List[CircleSpec]) -> List[Point]:
    x1, y1 = p1
    x2, y2 = p2
    # straight Ls
    via1, via2 = (x2, y1), (x1, y2)
    if _clear(p1, via1, obs) and _clear(via1, p2, obs):
        return [p1, via1, p2]
    if _clear(p1, via2, obs) and _clear(via2, p2, obs):
        return [p1, via2, p2]

    # iterative staircase
    offset = CLEARANCE
    steps  = 0
    while offset <= MAX_EXPAND and steps < MAX_STEPS:
        viaA = (x1 + offset, y1)
        viaB = (x1 + offset, y2)
        if _clear(p1, viaA, obs) and _clear(viaA, viaB, obs) and _clear(viaB, p2, obs):
            return [p1, viaA, viaB, p2]
        viaA = (x1 - offset, y1)
        viaB = (x1 - offset, y2)
        if _clear(p1, viaA, obs) and _clear(viaA, viaB, obs) and _clear(viaB, p2, obs):
            return [p1, viaA, viaB, p2]
        offset += CLEARANCE
        steps  += 1

    # fallback straight segment across discs
    return [p1, p2]


def euclidean_route(p1: Point, p2: Point,
                    obs: List[CircleSpec]) -> List[Point]:
    if _clear(p1, p2, obs):
        return [p1, p2]

    # first blocking circle
    for cx, cy, r0 in obs:
        if _intersects(p1, p2, (cx, cy, r0)):
            break
    else:
        print('AAAAAAAAAAAAAA')
        return [p1, p2]            # shouldn’t happen
    

    r = r0 + CLEARANCE
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    perp   = (-uy, ux), (uy, -ux)

    scale  = 1.0
    steps  = 0
    while scale * r <= MAX_EXPAND and steps < MAX_STEPS:
        for px, py in perp:
            via = (cx + px * r * scale, cy + py * r * scale)
            if _clear(p1, via, obs) and _clear(via, p2, obs):
                return [p1, via, p2]
        scale += 0.5
        steps += 1

    # ultimate fallback
    return [p1, p2]
