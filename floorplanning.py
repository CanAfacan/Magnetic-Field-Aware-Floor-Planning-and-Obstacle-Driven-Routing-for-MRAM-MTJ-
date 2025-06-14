# floorplanning.py  
"""
Placement engine 

Grey logic blocks may NEVER overlap each other or intrude the red
   MTJ danger discs (radius = fieldradius+MTJ_MARGIN).
Any move that breaks either rule is rejected immediately, so every state
   accepted by the annealer is always routable.
"""

from __future__ import annotations
import copy, math, random, time
from typing import Callable, Iterable, List, Tuple

# const
CanvasSize  = Tuple[int, int]

WIRE_PITCH  = 2.0          # ≈ one routing track (floor-units)
INIT_GAP    = 6.0          # starts annealing with generous gap
FINAL_GAP   = 4.0          # finishes with 2 to 3 tracks between blocks
MTJ_MARGIN  = 1.0          # extra padding around B-field radius
PENALTY     = 1e9          # weight for hard-constraint infractions


class Block:
    def __init__(self, block_id: str, w: float, h: float) -> None:
        self.id, self.w, self.h = block_id, w, h
        self.x = self.y = 0.0        # mutable origin (top-left)

    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def bbox(self, m: float = 0.0) -> Tuple[float, float, float, float]:
        
        return self.x - m, self.y - m, self.x + self.w + m, self.y + self.h + m

    def __getstate__(self):         # for deepcopy snapshot
        return self.id, self.w, self.h, self.x, self.y

    def __setstate__(self, st):
        self.id, self.w, self.h, self.x, self.y = st


# low-level geometry  
def _rects_overlap(a: Block, b: Block, gap: float) -> bool:
    ax1, ay1, ax2, ay2 = a.bbox(gap)
    bx1, by1, bx2, by2 = b.bbox(gap)
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _rect_circle_overlap(b: Block, cx: float, cy: float, r: float) -> bool:
    dx = max(abs(cx - (b.x + b.w / 2)) - b.w / 2, 0)
    dy = max(abs(cy - (b.y + b.h / 2)) - b.h / 2, 0)
    return dx * dx + dy * dy < r * r


def _violates_keepout(block: Block,
                      mtj_blocks: List[Block],
                      keep_out: List[Tuple[float, float, float]]) -> bool:
    """True if block intrudes any inflated MTJ circle."""
    if block in mtj_blocks:
        return False
    bx1, by1, bx2, by2 = block.bbox()
    for cx, cy, r in keep_out:
        # fast reject via AABB
        if bx2 < cx - r or bx1 > cx + r or by2 < cy - r or by1 > cy + r:
            continue
        if _rect_circle_overlap(block, cx, cy, r):
            return True
    return False


# back-compat helpers used by main.py
def blocks_overlap(a: Block, b: Block, margin: float = 0.0) -> bool:
    """Axis-aligned rectangle overlap with optional margin."""
    return _rects_overlap(a, b, margin)


def any_overlap(blocks: List[Block], margin: float = 0.0) -> bool:
    """True if ANY pair of blocks overlaps (with >= margin)."""
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if _rects_overlap(blocks[i], blocks[j], margin):
                return True
    return False


# cost function factory
def make_cost_fn(mtj_blocks: List[Block],
                 mtj_radii: List[float],
                 gap_target: float):
    """
    Returns a callable cost(blocks) and attaches helper attributes that the
    annealer will use for fast hardconstraint checks.
    """
    keep_out = [
        (m.center()[0], m.center()[1], r + MTJ_MARGIN)
        for m, r in zip(mtj_blocks, mtj_radii)
    ]

    def cost(blocks: Iterable[Block]) -> float:
        blks = list(blocks)

        # hard constraints enforced AGAIN inside cost 
        for i in range(len(blks)):
            for j in range(i + 1, len(blks)):
                if _rects_overlap(blks[i], blks[j], gap_target):
                    return PENALTY
        for b in blks:
            if _violates_keepout(b, mtj_blocks, keep_out):
                return PENALTY

        # soft objective = bounding area + void-slack between blocks
        xs = [b.x for b in blks]
        ys = [b.y for b in blks]
        xe = [b.x + b.w for b in blks]
        ye = [b.y + b.h for b in blks]
        area = (max(xe) - min(xs)) * (max(ye) - min(ys))

        slack = 0.0
        for i in range(len(blks)):
            for j in range(i + 1, len(blks)):
                dx = max(0, abs(blks[i].center()[0] - blks[j].center()[0])
                           - (blks[i].w + blks[j].w) / 2)
                dy = max(0, abs(blks[i].center()[1] - blks[j].center()[1])
                           - (blks[i].h + blks[j].h) / 2)
                gap = math.hypot(dx, dy)
                extra = max(0, gap - gap_target)
                slack += extra * extra
        return area + 0.05 * slack


    cost.keep_out    = keep_out
    cost.mtj_blocks  = mtj_blocks
    cost.gap_target  = gap_target
    return cost


# SA (generator)
def simulated_annealing(blocks: List[Block],
                        cost_fn: Callable[[Iterable[Block]], float],
                        init_layout: Callable[[List[Block]], None], 
                        *,
                        canvas_size: CanvasSize = (800, 600),
                        max_time: float = 60.0, # main.py passes 30
                        rng: random.Random | None = None):
                # Hard constraints 
    overlap_violation = any(_rects_overlap(blocks[i], blocks[j], cost_fn.gap_target) # uses cost_fn_to_pass.gap_target
                                   for i in range(len(blocks)) for j in range(i+1, len(blocks)))
    keepout_violation = any(_violates_keepout(b, cost_fn.mtj_blocks, cost_fn.keep_out) # uses cost_fn_to_pass attributes
                                   for b in blocks)
    #print("\n--- FLOORPLANNING: simulated_annealing marker---") 
    rng = rng or random.Random()
    W, H = canvas_size

    #print("FLOORPLAN_DEBUG: Calling init_layout function provided from main.py")
    init_layout(blocks) # re-applies 
    #print("FLOORPLAN_DEBUG: init_layout function call completed")

    cost_after_internal_init = cost_fn(blocks)
    #print(f"FLOORPLAN_DEBUG: Cost after internal init_layout call: {cost_after_internal_init}")
    if cost_after_internal_init >= PENALTY:
        print()
        #print("FLOORPLAN_DEBUG: !! PENALTY cost after internal init_layout call")


    T = 10.0
    T_min = 1e-3
    alpha = 0.90 
    iters = 4 * len(blocks) ** 2
    step = 0
    stage_gap = INIT_GAP # cost_fn.gap_target initially

    best_state = copy.deepcopy(blocks)
    best_cost = cost_fn(blocks) # Cost of the initial state
    curr_cost = best_cost
    #print(f"FLOORPLAN_DEBUG: Initial curr_cost = {curr_cost:.2f}, best_cost = {best_cost:.2f}")
    t0 = time.time()

    # the initial state, once
    yield step, blocks 

    while T > T_min and (time.time() - t0) < max_time:
        #print(f"FLOORPLAN_DEBUG: Temp loop: T={T:.4f}, Time Elapsed={time.time()-t0:.1f}s, Target Gap={cost_fn.gap_target:.1f}")
        amp = max(W, H) * 0.25 * T
        accepted_this_temp = 0
        hard_rejects_this_temp = 0

        for _ in range(iters):
            undo = _mutate(blocks, W, H, amp, rng)

            # Hard constraints 
            overlap_violation = any(_rects_overlap(blocks[i], blocks[j], cost_fn.gap_target)
                                   for i in range(len(blocks)) for j in range(i+1, len(blocks)))
            keepout_violation = any(_violates_keepout(b, cost_fn.mtj_blocks, cost_fn.keep_out)
                                   for b in blocks)

            if overlap_violation:
                # print("Hard reject - overlap") 
                undo(); hard_rejects_this_temp += 1; continue
            if keepout_violation:
                # print("Hard reject - keepout") 
                undo(); hard_rejects_this_temp += 1; continue
            
            new_cost = cost_fn(blocks)
            if new_cost >= PENALTY: # not happen if hard constraints above passed
                print()
                #print(f"FLOORPLAN_DEBUG: WARNING")


            dE = new_cost - curr_cost
            accept_prob = math.exp(-dE / T) if dE > 0 and T > 0 else 0 # to avoid domain error with T=0
            accept = dE <= 0 or rng.random() < accept_prob

            if accept:
                curr_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = copy.deepcopy(blocks)
                step += 1
                accepted_this_temp += 1
                yield step, blocks
            else:
                undo()
        
        print(f"FLOORPLAN_DEBUG: T={T:.4f}, Accepted={accepted_this_temp}/{iters}, Hard Rejects={hard_rejects_this_temp}")

        if T < 1.0 and stage_gap != FINAL_GAP:
            print(f"FLOORPLAN_DEBUG: Tightening gap. Old stage_gap: {stage_gap}, New stage_gap: {FINAL_GAP}")
            stage_gap = FINAL_GAP
            
            original_mtj_radii = []
            if cost_fn.keep_out:
                 original_mtj_radii = [r_val - MTJ_MARGIN for _, _, r_val in cost_fn.keep_out]
            
            # new base cost function
            new_base_cost_fn = make_cost_fn(cost_fn.mtj_blocks, # attributes from the current (possibly wrapped) cost_fn
                                            original_mtj_radii,
                                            stage_gap)
            
            # Checks here if the original cost_fn was for a wheel
            if hasattr(cost_fn, 'is_wheel_cost') and cost_fn.is_wheel_cost and hasattr(cost_fn, 'wheel_misalignment_func'):
                print("FLOORPLAN_DEBUG: Reconstructing WHEEL cost function with new base.")
                wheel_func_component = cost_fn.wheel_misalignment_func

                reconstructed_cost_fn = lambda blks_arg: new_base_cost_fn(blks_arg) + wheel_func_component(blks_arg)
                reconstructed_cost_fn.gap_target = new_base_cost_fn.gap_target
                reconstructed_cost_fn.keep_out   = new_base_cost_fn.keep_out
                reconstructed_cost_fn.mtj_blocks = new_base_cost_fn.mtj_blocks
                reconstructed_cost_fn.is_wheel_cost = True
                reconstructed_cost_fn.wheel_misalignment_func = wheel_func_component
                
                cost_fn = reconstructed_cost_fn
            else:
                print("FLOORPLAN_DEBUG: Reconstructing RECTANGULAR (or non-wheel) cost function.")
                cost_fn = new_base_cost_fn 
            
            curr_cost = cost_fn(blocks)
            best_cost = min(best_cost, curr_cost) 
            print(f"FLOORPLAN_DEBUG: New cost_fn applied. curr_cost={curr_cost:.2f}, best_cost={best_cost:.2f}, new gap_target={cost_fn.gap_target:.1f}")
            if curr_cost >= PENALTY:
                print("FLOORPLAN_DEBUG: PENALTY after gap tightening and new cost_fn.")


        T *= alpha

    print(f"FLOORPLAN_DEBUG: Annealing loop ended. T={T:.4f}, Time Elapsed={time.time()-t0:.1f}s")
    for src, dst in zip(best_state, blocks):
        dst.x, dst.y = src.x, src.y
    print(f"FLOORPLAN_DEBUG: Restored best state. Yielding final step {step + 1}")
    yield step + 1, blocks

# simple starter layouts 
def rectangular_layout(blocks: List[Block],
                       cols: int,
                       spacing: float,
                       margin: float = 10.0) -> None:
    max_w = max(b.w for b in blocks)
    max_h = max(b.h for b in blocks)
    cell_w = max_w + spacing
    cell_h = max_h + spacing
    for idx, blk in enumerate(blocks):
        r, c = divmod(idx, cols)
        blk.x = margin + c * cell_w + (cell_w - blk.w) / 2
        blk.y = margin + r * cell_h + (cell_h - blk.h) / 2


def minimum_wheel_radius(blocks: List[Block], gap: float = 10.0) -> float:
    perim = sum(math.hypot(b.w, b.h) + gap for b in blocks)
    return perim / (2.0 * math.pi)


def wheel_layout(blocks: List[Block],
                 center: Tuple[float, float],
                 radius: float | None = None,
                 gap: float = 10.0) -> None:
    if radius is None:
        radius = minimum_wheel_radius(blocks, gap)
    n = len(blocks)
    for i, blk in enumerate(blocks):
        theta = 2 * math.pi * i / n
        blk.x = center[0] + radius * math.cos(theta) - blk.w / 2
        blk.y = center[1] + radius * math.sin(theta) - blk.h / 2


# internal move kernel 
def _mutate(blocks: List[Block], W: float, H: float,
            amp: float, rng: random.Random):
    """Swap two blocks or jitter one; returns a lambda that undoes the move."""
    if rng.random() < 0.3:                # swap
        i, j = rng.sample(range(len(blocks)), 2)
        b1, b2 = blocks[i], blocks[j]
        old1, old2 = (b1.x, b1.y), (b2.x, b2.y)
        b1.x, b1.y, b2.x, b2.y = old2[0], old2[1], old1[0], old1[1]
        return lambda: (setattr(b1, "x", old1[0]), setattr(b1, "y", old1[1]),
                        setattr(b2, "x", old2[0]), setattr(b2, "y", old2[1]))
    else:                                 # jitter
        blk = rng.choice(blocks)
        old = (blk.x, blk.y)
        blk.x = min(max(blk.x + rng.uniform(-amp, amp), 0.0), W - blk.w)
        blk.y = min(max(blk.y + rng.uniform(-amp, amp), 0.0), H - blk.h)
        return lambda: (setattr(blk, "x", old[0]), setattr(blk, "y", old[1]))
    
