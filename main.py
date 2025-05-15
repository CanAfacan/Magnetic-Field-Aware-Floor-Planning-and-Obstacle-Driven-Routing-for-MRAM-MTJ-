# main.py
import os, sys, random, math, tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox # 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle
import time
import numpy as np 

try:
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import mtj_calc, floorplanning, wiring

sys.path.insert(0, os.path.dirname(__file__))

PIN_COLORS = plt.cm.tab20.colors


def _colour_menu(parent, var):
    #20 COLOURS WITH OPTION MENU as if they are metal layers, 20 is for just test, 10 will be used in last version
    menu = tk.OptionMenu(parent, var, *range(len(PIN_COLORS)))
    for i, item in enumerate(menu["menu"].children.values()):
        r, g, b = [int(255 * c) for c in PIN_COLORS[i][:3]]
        item.configure(background=f"#{r:02x}{g:02x}{b:02x}")
    return menu


class FloorplanApp:
    CANVAS_W, CANVAS_H = 800, 600

    # GUI INIT
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("MTJ‑Aware VLSI Floorplanner")
        master.resizable(True, True)

        self.all_routed_segments: list[tuple[tuple[float,float], tuple[float,float]]] = []
        

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # random blocks
        self.blocks: list[floorplanning.Block] = []
        for i in range(10):
            w, h = random.uniform(50, 100), random.uniform(50, 100)
            blk = floorplanning.Block(f"B{i}", w, h)
            while True:
                blk.x = random.uniform(0, self.CANVAS_W - w)
                blk.y = random.uniform(0, self.CANVAS_H - h)
                if not floorplanning.any_overlap(self.blocks + [blk]):
                    break
            self.blocks.append(blk)

        self.mtj_blocks = self.blocks[:2]
        self.mtj_params = {b.id: None for b in self.blocks}
        self.mtj_radii = []
        self.pin_specs = {}  # <<<<<<<<<<<<<<<<<<< 

        # Now calls _update_mtj_radii(), which calls draw()
        # draw() will find an empty self.pin_specs, which is fine for its loop.
        self._update_mtj_radii()

        sides = ["top", "bottom", "left", "right"]
        # self.pin_specs = {} 
        for b in self.blocks:
            if b in self.mtj_blocks:
                self.pin_specs[b.id] = [("left", 0.5, 0), ("right", 0.5, 0)]
            else:
                self.pin_specs[b.id] = [
                    (random.choice(sides), (i + 1) / 21, i % 20) for i in range(20)
                ]
        
        # control bar 
        bar = tk.Frame(master); bar.pack(fill=tk.X)

        tk.Button(bar, text="Blocks…", command=self.edit_blocks).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="MTJ Layers…", command=self.edit_mtj).pack(side=tk.LEFT)

        self.seed_var = tk.StringVar(value="rectangular")
        tk.Label(bar, text="Seed layout:").pack(side=tk.LEFT, padx=(10, 2))
        tk.OptionMenu(bar, self.seed_var, "rectangular", "wheel").pack(side=tk.LEFT)

        tk.Button(bar, text="Run Annealer",
                  command=self.start_floorplanning).pack(side=tk.LEFT, padx=4)

        self.route_var = tk.StringVar(value="Manhattan")
        tk.Label(bar, text="Routing:").pack(side=tk.LEFT, padx=(20, 2))
        tk.OptionMenu(bar, self.route_var, "Manhattan", "Euclidean").pack(side=tk.LEFT)

        tk.Button(bar, text="Route Nets", command=self.start_routing).pack(side=tk.LEFT, padx=4)

        # Button for E-Field Map
        self.efield_button = tk.Button(bar, text="Show E‑Field Map",
                                       command=self.show_efield_map, state=tk.DISABLED)
        self.efield_button.pack(side=tk.LEFT, padx=4)


        self.canvas.mpl_connect("button_press_event",  self._on_mouse_down)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_up)

        self.draw()

    def _disable_efield_button_and_clear_routes(self):
        """Helper to reset routing state and disable E-field map button."""
        if hasattr(self, 'efield_button'): 
            self.efield_button.config(state=tk.DISABLED)
        self.all_routed_segments.clear()
        self.draw()
        if self.master.winfo_exists():
             self.master.update_idletasks()


    # helpers & state
    def _update_mtj_radii(self):
        self._disable_efield_button_and_clear_routes() # MTJ radii change invalidates routing
        self.mtj_radii = []
        for b in self.mtj_blocks:
            dev = self.mtj_params[b.id]
            self.mtj_radii.append(dev.field_zone_radius(area=b.w * b.h) if dev else 0.0)

    # drawing routine 
    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.CANVAS_W)
        self.ax.set_ylim(0, self.CANVAS_H)

        # MTJ danger zones
        # Checks if mtj_radii exists (it should by now due to __init__ ordering)
        if hasattr(self, 'mtj_radii'):
            for b, r in zip(self.mtj_blocks, self.mtj_radii):
                cx, cy = b.center()
                self.ax.add_patch(Circle((cx, cy), r, facecolor="red",
                                         alpha=0.2, edgecolor=None, zorder=0))

        for b in self.blocks: # self.blocks should exist
            if b in self.mtj_blocks:
                R = min(b.w, b.h) / 2
                cx, cy = b.center()
                self.ax.add_patch(Circle((cx, cy), R,
                                         facecolor="lightcoral", edgecolor="k", zorder=1))
                self.ax.text(cx, cy, b.id, ha="center", va="center", zorder=2)
            else:
                self.ax.add_patch(Rectangle((b.x, b.y), b.w, b.h,
                                            facecolor="lightgrey", edgecolor="k", zorder=1))
                self.ax.text(b.x + 2, b.y + 2, b.id, zorder=2)

            # pindrawings robustness check
            # Checks if pin_specs attribute exists AND if the current block's ID is a key in pin_specs
            if hasattr(self, 'pin_specs') and b.id in self.pin_specs: 
                for idx, pin in enumerate(self.pin_specs[b.id]):
                    if len(pin) == 3:
                        side, frac, col = pin
                    else: # legacy two‑tuple
                        side, frac = pin
                        col = idx % 20
                    colour = PIN_COLORS[col]
                    if side == "top":
                        px, py = b.x + frac * b.w, b.y + b.h
                    elif side == "bottom":
                        px, py = b.x + frac * b.w, b.y
                    elif side == "left":
                        px, py = b.x, b.y + frac * b.h
                    else: # right
                        px, py = b.x + b.w, b.y + frac * b.h
                    self.ax.plot(px, py, "o", color=colour, ms=6, zorder=3)

        if hasattr(self, 'canvas'):
            self.canvas.draw()

    # drag / drop (blocks)
    def _hit_test(self, event): 
        if event.inaxes != self.ax or event.xdata is None:
            return None
        x, y = event.xdata, event.ydata
        for b in reversed(self.blocks):  # top‑most first
            if b in self.mtj_blocks:
                cx, cy = b.center()
                if (x - cx) ** 2 + (y - cy) ** 2 <= (min(b.w, b.h) / 2) ** 2:
                    return b
            else:
                if b.x <= x <= b.x + b.w and b.y <= y <= b.y + b.h:
                    return b
        return None

    def _on_mouse_down(self, event):
        blk = self._hit_test(event)
        if not blk:
            return
        self.drag_block = blk
        self.drag_dx = event.xdata - blk.x
        self.drag_dy = event.ydata - blk.y

    def _on_mouse_move(self, event):
        if not hasattr(self, "drag_block") or event.inaxes != self.ax:
            return
        # self._disable_efield_button_and_clear_routes() 
        blk = self.drag_block
        blk.x = max(0, min(event.xdata - self.drag_dx, self.CANVAS_W - blk.w))
        blk.y = max(0, min(event.ydata - self.drag_dy, self.CANVAS_H - blk.h))
        
        
        if hasattr(self, "_disable_efield_button_and_clear_routes"):
            self._disable_efield_button_and_clear_routes() # calls self.draw()
        else:
            self.draw() # Fallback if the helper isn't there for some reason


    def _on_mouse_up(self, _):
        if hasattr(self, "drag_block"):
            del self.drag_block


    # block / pin management dialogs 
    def edit_blocks(self):
        self._disable_efield_button_and_clear_routes() # Block changes invalidate routes
        dlg = tk.Toplevel(self.master); dlg.title("Blocks")
        lst = tk.Listbox(dlg, width=22); lst.grid(row=0, column=0, rowspan=6, sticky="ns")

        def refresh():
            lst.delete(0, tk.END)
            for b in self.blocks:
                lst.insert(tk.END, f"{b.id}  ({int(b.w)}×{int(b.h)})") # <<< Should be b.h?
        #refresh()

        refresh()

        def on_change_and_redraw():
            self._disable_efield_button_and_clear_routes()
            refresh()
            self.draw()

        def add_block():
            bid = f"B{len(self.blocks)}"
            w = simpledialog.askfloat("Width", "Width?", minvalue=10, parent=dlg)
            h = simpledialog.askfloat("Height", "Height?", minvalue=10, parent=dlg)
            if not w or not h:
                return
            blk = floorplanning.Block(bid, w, h)
            blk.x, blk.y = 10, 10
            self.blocks.append(blk)
            self.pin_specs[blk.id] = []
            if blk.id not in self.mtj_params: # to make sure that the new blocks have MTJ params entry
                self.mtj_params[blk.id] = None
            on_change_and_redraw()


        def delete_block():
            sel = lst.curselection()
            if not sel:
                return
            blk = self.blocks.pop(sel[0])
            if blk in self.mtj_blocks:
                self.mtj_blocks.remove(blk)
            del self.pin_specs[blk.id]
            if blk.id in self.mtj_params:
                del self.mtj_params[blk.id]
            on_change_and_redraw()

        def resize_block():
            sel = lst.curselection()
            if not sel:
                return
            blk = self.blocks[sel[0]]
            w = simpledialog.askfloat("Width", "New width?", initialvalue=blk.w, parent=dlg)
            h = simpledialog.askfloat("Height", "New height?", initialvalue=blk.h, parent=dlg)
            if w and h:
                blk.w, blk.h = w, h
                on_change_and_redraw()

        def edit_pins():
            sel = lst.curselection()
            if not sel:
                return
            blk = self.blocks[sel[0]]
            self._pin_editor(blk) # Pin editor itself calls self.draw() upon OK

        tk.Button(dlg, text="➕ Block", command=add_block).grid(row=0, column=1, sticky="ew")
        tk.Button(dlg, text="Resize", command=resize_block).grid(row=1, column=1, sticky="ew")
        tk.Button(dlg, text="Delete", command=delete_block).grid(row=2, column=1, sticky="ew")
        tk.Button(dlg, text="Pins…", command=edit_pins).grid(row=3, column=1, sticky="ew")

    def _pin_editor(self, blk):
        # Pin changes don't strictly require re-routing if only colors change
        # But if positions/sides change, it does. Simpler to always invalidate.
        self._disable_efield_button_and_clear_routes()
        dlg = tk.Toplevel(self.master); dlg.title(f"Pins – {blk.id}")
        pins = list(self.pin_specs[blk.id])

        lst = tk.Listbox(dlg, width=22); lst.grid(row=0, column=0, rowspan=5)

        def redraw_pin_list(): 
            lst.delete(0, tk.END)
            for side, frac, col in pins:
                colour = PIN_COLORS[col]
                lst.insert(tk.END, f"{side}@{frac:.2f}")
                lst.itemconfig(tk.END, fg=f"#{int(255*colour[0]):02x}{int(255*colour[1]):02x}{int(255*colour[2]):02x}")

        redraw_pin_list()

        side_var = tk.StringVar(value="top")
        frac_var = tk.DoubleVar(value=0.5)
        col_var  = tk.IntVar(value=0)

        tk.OptionMenu(dlg, side_var, "top", "bottom", "left", "right").grid(row=0, column=1, sticky="ew")
        tk.Scale(dlg, from_=0, to=1, resolution=0.05, orient="horizontal",
                 variable=frac_var, label="Offset").grid(row=1, column=1, sticky="ew")
        _colour_menu(dlg, col_var).grid(row=2, column=1, sticky="ew")

        def add_pin():
            pins.append((side_var.get(), frac_var.get(), col_var.get()))
            redraw_pin_list()
        tk.Button(dlg, text="Add", command=add_pin).grid(row=3, column=1, sticky="ew")

        def del_pin():
            sel = lst.curselection()
            if sel:
                pins.pop(sel[0]); redraw_pin_list()
        tk.Button(dlg, text="Delete", command=del_pin).grid(row=4, column=1, sticky="ew")

        def on_ok():
            self.pin_specs[blk.id] = pins
            self._disable_efield_button_and_clear_routes() 
            self.draw() 
            dlg.destroy()
        tk.Button(dlg, text="OK", command=on_ok).grid(row=5, column=0, columnspan=2, sticky="ew")


    # MTJ settings dialog 
    def edit_mtj(self):
        # _disable_efield_button_and_clear_routes() is called by _update_mtj_radii
        dlg = tk.Toplevel(self.master); dlg.title("MTJ Properties")
        entries = {}
        row = 0
        for b in self.mtj_blocks:
            tk.Label(dlg, text=b.id, fg="blue").grid(row=row, column=0, columnspan=4); row += 1
            for name in ("free", "barrier", "reference", "fixed", "antiferro"):
                tk.Label(dlg, text=f"{name} (nm)").grid(row=row, column=0)
                e1 = tk.Entry(dlg, width=10); e1.grid(row=row, column=1) # Adjusted width
                tk.Label(dlg, text="Ms (A/m)").grid(row=row, column=2)
                e2 = tk.Entry(dlg, width=10); e2.grid(row=row, column=3) # Adjusted width
                prev = self.mtj_params.get(b.id) # Use .get for safety
                if prev:
                    for lay in prev.layers:
                        if lay["name"] == name:
                            e1.insert(0, str(lay["thickness"] * 1e9))
                            e2.insert(0, str(lay["Ms"]))
                entries[(b.id, name)] = (e1, e2); row += 1

        def on_ok():
            for b in self.mtj_blocks:
                layers = []
                for name in ("free", "barrier", "reference", "fixed", "antiferro"):
                    try:
                        t_str = entries[(b.id, name)][0].get()
                        ms_str = entries[(b.id, name)][1].get()
                        t = float(t_str) * 1e-9 if t_str else 0.0
                        Ms = float(ms_str) if ms_str else 0.0
                    except ValueError:
                        messagebox.showerror("Input Error", f"Invalid number for {name} in {b.id}", parent=dlg)
                        return
                    layers.append({"name": name, "thickness": t, "Ms": Ms})
                self.mtj_params[b.id] = mtj_calc.MTJDevice(layers)
            self._update_mtj_radii() # This calls _disable_efield_button_and_clear_routes
            dlg.destroy()
            self.draw()

        tk.Button(dlg, text="OK", command=on_ok).grid(row=row, column=0, columnspan=4)

    #  placement / annealing 
    def _wheel_misalignment(self, blocks, center, radius): # FloorplanApp
        err, n = 0.0, len(blocks)
        for i, b in enumerate(blocks):
            theta = 2 * math.pi * i / n
            tx = center[0] + radius * math.cos(theta)
            ty = center[1] + radius * math.sin(theta)
            cx, cy = b.center()
            err += (cx - tx) ** 2 + (cy - ty) ** 2
        return err / n if n > 0 else 0.0 # Added n > 0 check

    def start_floorplanning(self):
        print("\n STARTING FLOORPLANNING ") # More prominent marker
        self._disable_efield_button_and_clear_routes()
        style = self.seed_var.get()
        #print(f"MAIN_DEBUG: Seed style: {style}")
       # print(f"MAIN_DEBUG: Num blocks: {len(self.blocks)}, MTJ blocks: {len(self.mtj_blocks)}, MTJ radii: {self.mtj_radii}")

        if not self.blocks:
            messagebox.showinfo("Annealer Info", "No blocks to floorplan.", parent=self.master)
            self.draw(); self.master.update()
            return

        # Initial Layout Parameters
        cols_for_rect = math.ceil(math.sqrt(len(self.blocks)))
        spacing_for_rect = 25.0 
        margin_for_rect = 30.0   
        gap_for_wheel = 15.0    

        #print(f"MAIN_DEBUG: Rectangular layout params: cols={cols_for_rect}, spacing={spacing_for_rect}, margin={margin_for_rect}")
        #print(f"MAIN_DEBUG: Wheel layout params: gap={gap_for_wheel}")
        #print(f"MAIN_DEBUG: Annealer INIT_GAP={floorplanning.INIT_GAP}, FINAL_GAP={floorplanning.FINAL_GAP}")

        cost_fn_to_pass = None 
        wheel_misalignment_component = None # to store the function or its result

        if style.lower() == "wheel":
            center = (self.CANVAS_W / 2, self.CANVAS_H / 2)
            radius = floorplanning.minimum_wheel_radius(self.blocks, gap=gap_for_wheel)
            #print(f"MAIN_DEBUG: Wheel radius: {radius}")
            init_layout_func = lambda blks: floorplanning.wheel_layout(blks, center, radius, gap=gap_for_wheel)
            
            base_cost_fn_wheel = floorplanning.make_cost_fn(
                self.mtj_blocks, self.mtj_radii, floorplanning.INIT_GAP
            )
            
            # Define the misalignment function separately
            #
            #
            #

            def actual_wheel_misalignment_func(blks_arg): 
                return 0.1 * self._wheel_misalignment(blks_arg, center, radius)

            # The combined cost function
            cost_fn_to_pass = lambda blks_arg: base_cost_fn_wheel(blks_arg) + actual_wheel_misalignment_func(blks_arg)
            
            cost_fn_to_pass.gap_target = base_cost_fn_wheel.gap_target
            cost_fn_to_pass.keep_out   = base_cost_fn_wheel.keep_out
            cost_fn_to_pass.mtj_blocks = base_cost_fn_wheel.mtj_blocks
            # Store the component function for reconstruction
            cost_fn_to_pass.wheel_misalignment_func = actual_wheel_misalignment_func 
            cost_fn_to_pass.is_wheel_cost = True # Flag
            print("MAIN_DEBUG: WHEEL")

        else: # Rectangular
            init_layout_func = lambda blks: floorplanning.rectangular_layout(
                blks, cols=cols_for_rect, spacing=spacing_for_rect, margin=margin_for_rect
            )
            cost_fn_to_pass = floorplanning.make_cost_fn(
                self.mtj_blocks, self.mtj_radii, floorplanning.INIT_GAP
            )
            print("MAIN_DEBUG: RECTANGULAR")
        
        init_layout_func(self.blocks) #  initial layout
        
        # checking initial layout against hard constraints 
        initial_cost_val = cost_fn_to_pass(self.blocks)
        #print(f"MAIN_DEBUG: Cost of initial layout: {initial_cost_val}")
        if initial_cost_val >= floorplanning.PENALTY:
            #print("MAIN_DEBUG: WARNING! Initial layout results in PENALTY cost.")
            # Further check which constraint is violated by the initial layout
            if any(floorplanning._rects_overlap(self.blocks[i], self.blocks[j], cost_fn_to_pass.gap_target)
                   
                   for i in range(len(self.blocks)) for j in range(i+1, len(self.blocks))):
                print()
                #print("MAIN_DEBUG: Initial layout has OVERLAPPING blocks.")
            if any(floorplanning._violates_keepout(b, cost_fn_to_pass.mtj_blocks, cost_fn_to_pass.keep_out)
                   for b in self.blocks):
                print()
                #print("MAIN_DEBUG: Initial layout VIOLATES MTJ KEEPOUT.")
        # init layout chekc ends

        self.draw() # initial layout
        self.master.update()
        #print("MAIN_DEBUG: Initial layout drawn. Starting simulated_annealing loop...")
        
        anneal_iterations_yielded = 0
        max_anneal_time = 30 # s

        # Pass init_layout_func to the annealer.
        # The annealer itself calls init_layout(blocks) internally.
        # So the one above is just for our pre-check and initial draw.
        for iter_num, _ in floorplanning.simulated_annealing(
            self.blocks, cost_fn_to_pass, init_layout_func, 
            canvas_size=(self.CANVAS_W, self.CANVAS_H), max_time=max_anneal_time):
            
            anneal_iterations_yielded += 1
            if anneal_iterations_yielded % 50 == 0 or anneal_iterations_yielded == 1:
                 
                 print()
                 #
                 # print(f"MAIN_DEBUG: Annealer yielded (main count: {anneal_iterations_yielded}, annealer step: {iter_num})")
            
            self.draw()
            self.master.update()
        
        #print(f"MAIN_DEBUG: Annealing loop finished total yields received: {anneal_iterations_yielded}")

        if self.blocks:
            xs, ys = [b.x for b in self.blocks], [b.y for b in self.blocks]
            xe = [b.x + b.w for b in self.blocks]
            ye = [b.y + b.h for b in self.blocks]
            self.ax.add_patch(
                Rectangle((min(xs), min(ys)), max(xe) - min(xs), max(ye) - min(ys),
                          fill=False, edgecolor="blue", lw=2, zorder=4)
            )
        self.canvas.draw()
        print("MAIN_DEBUG: Final bounding box drawn.")

    # wiring / routing with delay animations so that it looks cool
    def start_routing(self, delay: float = 0.0004):
        self._disable_efield_button_and_clear_routes() # clears the previous state

        mode = self.route_var.get().lower()

        max_pins = 0
        if self.blocks: # are there blocks before accessing pin_specs?
             max_pins = max(len(self.pin_specs.get(b.id, [])) for b in self.blocks) if self.pin_specs else 0


        nets: list[tuple[int, list[tuple[floorplanning.Block, tuple[float, float]]]]] = []

        for idx in range(max_pins):
            pins = []
            for blk in self.blocks:
                block_pins = self.pin_specs.get(blk.id, [])
                if idx >= len(block_pins):
                    continue
                side, frac, *_ = block_pins[idx]
                if side == "top":
                    pt = (blk.x + frac * blk.w, blk.y + blk.h)
                elif side == "bottom":
                    pt = (blk.x + frac * blk.w, blk.y)
                elif side == "left":
                    pt = (blk.x, blk.y + frac * blk.h)
                else:  # right
                    pt = (blk.x + blk.w, blk.y + frac * blk.h)
                pins.append((blk, pt))
            if len(pins) > 1:
                nets.append((idx, pins))

        mtj_disc = {blk: (*blk.center(), r)
                    for blk, r in zip(self.mtj_blocks, self.mtj_radii)}

        for net_idx, pins_on_net in nets: 
            colour = PIN_COLORS[net_idx % 20]
            for (blk_a, pa), (blk_b, pb) in zip(pins_on_net, pins_on_net[1:]):
                obstacles = [disc for mblk, disc in mtj_disc.items()
                             if mblk not in (blk_a, blk_b)]

                if mode == "manhattan":
                    path = wiring.manhattan_route(pa, pb, obstacles)
                else:
                    path = wiring.euclidean_route(pa, pb, obstacles)

                if path and len(path) > 1:
                    for p, q in zip(path, path[1:]):
                        self.all_routed_segments.append((p, q)) # <<< STORE SEGMENT
                        self.ax.plot((p[0], q[0]), (p[1], q[1]),
                                     "-", color=colour, zorder=2)
                        if delay > 0: # Only delay if requested for animation
                            self.canvas.draw_idle()
                            self.master.update_idletasks()
                            time.sleep(delay)
        
        if delay == 0: # If no delay, does a single final draw
            self.canvas.draw()
        elif not self.all_routed_segments: # If delay was >0 but no segments, still draws
             self.canvas.draw()


        if self.all_routed_segments:
            self.efield_button.config(state=tk.NORMAL)
        else:
            self.efield_button.config(state=tk.DISABLED)
            if not nets:
                messagebox.showinfo("Routing Info", "No nets to route (not enough pins defined or blocks).", parent=self.master)
            else:
                messagebox.showinfo("Routing Info", "Routing completed, but no valid paths found or drawn.", parent=self.master)


  
 # E-Field Map PLOTS
    def show_efield_map(self):
        if not self.all_routed_segments:
            messagebox.showinfo("Info", "No routing data available. Please run 'Route Nets' first.", parent=self.master)
            return

        GRID_CELL_SIZE = 10  
        GAUSSIAN_SIGMA = 2.5 
                             

        num_cells_x = int(math.ceil(self.CANVAS_W / GRID_CELL_SIZE))
        num_cells_y = int(math.ceil(self.CANVAS_H / GRID_CELL_SIZE))

        density_map = np.zeros((num_cells_y, num_cells_x))

        for p1, p2 in self.all_routed_segments:
            x1, y1 = p1
            x2, y2 = p2
            length_s = math.hypot(x2 - x1, y2 - y1)
            if length_s == 0:
                continue

            # roughly every half cell size, ensures at least 1 sample
            num_samples = max(1, int(math.ceil(length_s / (GRID_CELL_SIZE * 0.5))))
            
            density_per_sample = length_s / num_samples

            for i in range(num_samples):
                t = (i + 0.5) / num_samples # at center of sub-segment
                pt_x = x1 + t * (x2 - x1)
                pt_y = y1 + t * (y2 - y1)
                
                cell_x = int(pt_x / GRID_CELL_SIZE)
                cell_y = int(pt_y / GRID_CELL_SIZE)

                # Clamp to grid boundaries
                cell_x = max(0, min(cell_x, num_cells_x - 1))
                cell_y = max(0, min(cell_y, num_cells_y - 1))
                
                density_map[cell_y, cell_x] += density_per_sample


        density_map_to_plot = density_map # Default if no smoothing
        if SCIPY_AVAILABLE and np.any(density_map > 0): # Only applies filter if scipy is there and data exists
            try:
                # mode='constant', cval=0.0 handles edges by padding with zeros,
                density_map_to_plot = gaussian_filter(density_map, sigma=GAUSSIAN_SIGMA, mode='constant', cval=0.0)
            except Exception as e:
                print(f"Error during Gaussian smoothing: {e}. Using raw density map.")
                density_map_to_plot = density_map 
        elif not SCIPY_AVAILABLE:
             # Thi is now inside the mock gaussian_filter if SCIPY_AVAILABLE is 0

             pass


        x_centers = np.arange(num_cells_x) * GRID_CELL_SIZE + GRID_CELL_SIZE / 2
        y_centers = np.arange(num_cells_y) * GRID_CELL_SIZE + GRID_CELL_SIZE / 2
        X_mesh, Y_mesh = np.meshgrid(x_centers, y_centers)

        fig_3d = plt.figure(figsize=(12, 9)) 
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        surf = ax_3d.plot_surface(X_mesh, Y_mesh, density_map_to_plot, cmap='viridis', 
                               edgecolor='none', antialiased=True, rstride=1, cstride=1, shade=True)
        fig_3d.colorbar(surf, shrink=0.5, aspect=10, label="E-Field Intensity (Wire Density Proxy)")

        z_min_plot = np.min(density_map_to_plot) if density_map_to_plot.size > 0 else 0
        z_max_plot = np.max(density_map_to_plot) if density_map_to_plot.size > 0 else 1
        if abs(z_max_plot - z_min_plot) < 1e-6: # If field is flat

            z_max_plot = z_min_plot + max(1.0, 0.1 * abs(z_min_plot) if z_min_plot !=0 else 1.0) 


        # MTJ columns
        active_mtj_blocks_plot = [b for b in self.mtj_blocks if b.w > 0 and b.h > 0]
        for b in active_mtj_blocks_plot:
            cx, cy = b.center()
            R_mtj = min(b.w, b.h) / 2 
            if R_mtj <= 0: continue # zero-radius MTJs are skipped

            u_cyl = np.linspace(0, 2 * np.pi, 30)  
            z_cyl_coords = np.linspace(z_min_plot, z_max_plot, 2)

            U_cyl, Z_cyl_mesh = np.meshgrid(u_cyl, z_cyl_coords) 
            X_cyl = cx + R_mtj * np.cos(U_cyl)
            Y_cyl = cy + R_mtj * np.sin(U_cyl)
            

            ax_3d.plot_surface(X_cyl, Y_cyl, Z_cyl_mesh, color='deepskyblue', alpha=0.35, 
                            rstride=1, cstride=1, linewidth=0.1, antialiased=True, edgecolor='lightskyblue', shade=False)

        ax_3d.set_xlabel("X (floorplan units)")
        ax_3d.set_ylabel("Y (floorplan units)")
        ax_3d.set_zlabel("E-Field Intensity (arb. units)")
        ax_3d.set_title("3D E-Field Intensity Map (Wire Density Proxy)")

        ax_3d.view_init(elev=25, azim=-75) 

        if z_max_plot > 0 and z_max_plot > z_min_plot : 
             ax_3d.set_zlim(z_min_plot, z_max_plot * 1.05) 

        fig_3d.tight_layout()
        plt.show() 







# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FloorplanApp(root) 
        root.mainloop()
    except Exception as e:
        print("AN ERROR OCCURRED:")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit") 
