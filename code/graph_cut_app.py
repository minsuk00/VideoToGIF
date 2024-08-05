import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
from typing import Literal
import numpy as np

from code.grab_cut import GrabCut
from code.grab_cut_opencv import GrabCutOpenCV
from code.simulated_annealing import SimulatedAnnealing
from code.graph_cut import GraphCut


class GraphCutApp:
    def __init__(self, root, video_player):
        self.root = root
        self.video_player = video_player
        self.width = video_player.video_width
        self.height = video_player.video_height

        # ===================== BUTTONS =====================
        button_frame = tk.Frame(root)
        button_frame.pack()

        # Take snapshot button
        self.snapshot_button = tk.Button(button_frame, text="Take Snapshot", command=self.take_snapshot)
        # self.snapshot_button.pack(side=tk.TOP, fill=tk.X)
        self.snapshot_button.grid(row=0, column=0)

        # Process button
        self.process_button = tk.Button(
            button_frame, text="Process Image", command=self.process_image, state=tk.DISABLED
        )
        # self.process_button.pack(side=tk.TOP, fill=tk.X)
        self.process_button.grid(row=0, column=1)

        self.draw_mode_var = tk.BooleanVar(value=False)
        self.toggle_button_draw_mode = tk.Checkbutton(
            button_frame, text="Rectangle", variable=self.draw_mode_var, command=self.toggle_draw_mode
        )
        self.toggle_button_draw_mode.grid(row=0, column=2)

        self.brush_var = tk.BooleanVar(value=True)
        self.toggle_button_fg_bg_brush = tk.Checkbutton(
            button_frame, text="Foreground Brush", variable=self.brush_var, command=self.toggle_brush, state=tk.DISABLED
        )
        self.toggle_button_fg_bg_brush.grid(row=0, column=3)

        # Smoothness term scale
        smoothness_frame = tk.Frame(button_frame)
        smoothness_frame.grid(row=0, column=4)
        smoothness_scale_label = tk.Label(smoothness_frame, text="Smoothness Term Scale")
        smoothness_scale_label.pack(side=tk.BOTTOM)
        self.smoothness_scale = tk.Scale(
            smoothness_frame,
            from_=0.0,
            to=30.0,
            orient="horizontal",
            resolution=0.1,
            command=self.on_smoothness_slider_change,
        )
        self.smoothness_scale.set(1.0)  # Set initial value
        self.smoothness_scale.pack()

        # Data term scale
        data_frame = tk.Frame(button_frame)
        data_frame.grid(row=0, column=5)
        data_scale_label = tk.Label(data_frame, text="Data Term Scale")
        data_scale_label.pack(side=tk.BOTTOM)
        self.data_scale = tk.Scale(
            data_frame,
            from_=0.0,
            to=30.0,
            orient="horizontal",
            resolution=0.1,
            command=self.on_data_slider_change,
        )
        self.data_scale.set(1.0)  # Set initial value
        self.data_scale.pack()

        self.apply_explicit_mask_var = tk.BooleanVar(value=False)
        self.toggle_apply_explicit_mask = tk.Checkbutton(
            button_frame,
            text="Apply Explicit Mask",
            variable=self.apply_explicit_mask_var,
            state=tk.NORMAL,
        )
        self.toggle_apply_explicit_mask.grid(row=0, column=6)
        # ==================================================

        # ===================== CANVAS =====================
        canvas_frame = tk.Frame(root)
        canvas_frame.pack(pady=20)
        # Canvas for displaying snapshot and drawing
        self.canvas_input = tk.Canvas(
            canvas_frame,
            width=self.width,
            height=self.height,
            bg="black",
            highlightthickness=0,
        )
        # self.canvas_input.pack(side=tk.LEFT, padx=10)
        self.canvas_input.grid(row=0, column=0, padx=5)
        title_label_input = tk.Label(canvas_frame, text="Input", font=("Arial", 16))
        title_label_input.grid(row=1, column=0)

        self.canvas_output = tk.Canvas(
            canvas_frame,
            width=self.width,
            height=self.height,
            bg="black",
            # bg="white",
            highlightthickness=0,
        )
        # self.canvas_output.pack(side=tk.LEFT, padx=10)
        self.canvas_output.grid(row=0, column=1, padx=5)
        title_label_output = tk.Label(canvas_frame, text="Output", font=("Arial", 16))
        title_label_output.grid(row=1, column=1)

        self.canvas_input_image = None
        self.canvas_output_image = None
        self.canvas_image_np = None

        # Bind mouse events to canvas
        self.canvas_input.bind("<Button-1>", self.on_mouse_down)
        self.canvas_input.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas_input.bind("<ButtonRelease-1>", self.on_mouse_up)
        # =================================================

        # ===================== DRAWING =====================
        self.current_phase: Literal["draw", "process-image", "null"] = "null"
        self.start_x = None
        self.start_y = None
        self.current_item = None

        # Lists to keep track of the drawn shapes
        self.draw_mode: Literal["rectangle", "lines"] = "rectangle"
        self.rectangle = None
        self.line_masks = {
            "fg": np.zeros((self.height, self.width), dtype=np.uint8),
            "bg": np.zeros((self.height, self.width), dtype=np.uint8),
        }
        self.mode: Literal["fg", "bg"] = "fg"
        self.brush_size: int = 2
        self.color: Literal["red", "lime"] = "lime"
        self.graph_cut = None

    def on_smoothness_slider_change(self, value):
        if self.graph_cut:
            self.graph_cut.smoothness_term_scale = float(value)

    def on_data_slider_change(self, value):
        if self.graph_cut:
            self.graph_cut.data_term_scale = float(value)

    def toggle_draw_mode(self):
        if self.draw_mode_var.get():
            self.draw_mode = "lines"
            self.toggle_button_draw_mode.config(text="Lines")
            self.toggle_button_fg_bg_brush.config(state=tk.NORMAL)
        else:
            self.draw_mode = "rectangle"
            self.toggle_button_draw_mode.config(text="Rectangle")
            self.toggle_button_fg_bg_brush.config(state=tk.DISABLED)

        # self.process_button.config(state=tk.DISABLED)
        # self.line_masks = {
        #     "fg": np.zeros((self.height, self.width), dtype=np.uint8),
        #     "bg": np.zeros((self.height, self.width), dtype=np.uint8),
        # }
        # self.canvas_output.delete("all")
        self.current_phase = "draw"
        # if self.canvas_input_image:
        #     self.canvas_input.create_image(0, 0, anchor=tk.NW, image=self.canvas_input_image)

    def toggle_brush(self):
        if self.brush_var.get():
            self.mode = "fg"
            self.color = "lime"
            self.toggle_button_fg_bg_brush.config(text="Foreground Brush")
        else:
            self.mode = "bg"
            self.color = "red"
            self.toggle_button_fg_bg_brush.config(text="Background Brush")

    def update_mask(self, x, y):
        x0, y0, x1, y1 = x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size
        for i in range(max(0, y0), min(self.height, y1 + 1)):
            for j in range(max(0, x0), min(self.width, x1 + 1)):
                self.line_masks[self.mode][i][j] = 1

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.current_phase != "draw":
            return
        if self.draw_mode == "rectangle":
            if self.current_item:
                self.canvas_input.delete(self.current_item)
            self.current_item = self.canvas_input.create_rectangle(
                self.start_x, self.start_y, event.x, event.y, outline="cyan"
            )
        elif self.draw_mode == "lines":
            self.current_item = self.canvas_input.create_rectangle(
                self.start_x, self.start_y, event.x, event.y, outline="black"
            )
            pass

    def on_mouse_drag(self, event):
        if self.current_item:
            if self.current_phase != "draw":
                return
            if self.draw_mode == "rectangle":
                self.canvas_input.coords(self.current_item, self.start_x, self.start_y, event.x, event.y)
            elif self.draw_mode == "lines":
                self.canvas_input.create_oval(
                    event.x - self.brush_size,
                    event.y - self.brush_size,
                    event.x + self.brush_size,
                    event.y + self.brush_size,
                    fill=self.color,
                    outline=self.color,
                )
                self.update_mask(event.x, event.y)

    def on_mouse_up(self, event):
        if self.current_item:
            if self.current_phase != "draw":
                return
            if self.draw_mode == "rectangle":
                self.rectangle = [int(val) for val in self.canvas_input.coords(self.current_item)]
                # self.current_phase = "process-image"
                print(self.canvas_input.coords(self.current_item))
            elif self.draw_mode == "lines":
                self.process_button.config(state=tk.NORMAL)

    def take_snapshot(self):
        photo, np_photo = self.video_player.capture_current_frame()
        if photo:
            # keep reference in order to prevent gc
            self.canvas_input_image = photo
            self.canvas_image_np = np_photo  # (h x w x c)

            # if self.canvas_image_id is not None:
            #     self.canvas.delete(self.canvas_image_id)
            # self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_input.create_image(0, 0, anchor=tk.NW, image=photo)
            # self.canvas.create_image(0, 0, image=photo)

            ## Reset
            self.process_button.config(state=tk.DISABLED)
            self.current_phase = "draw"
            self.line_masks = {
                "fg": np.zeros((self.height, self.width), dtype=np.uint8),
                "bg": np.zeros((self.height, self.width), dtype=np.uint8),
            }
            self.canvas_output.delete("all")
        else:
            print("Error. Failed to take snapshot from video.")

    def process_image(self):
        # self.current_phase = "process-image"
        # if self.current_phase == "process-image":
        # if self.draw_mode == "rectangle":
        #     self.graph_cut = GraphCut(
        #         self.canvas_image_np,
        #         rect=self.rectangle,
        #         data_term_scale=self.data_scale.get(),
        #         smoothness_term_scale=self.smoothness_scale.get(),
        #     )
        # elif self.draw_mode == "lines":
        #     self.graph_cut = GraphCut(
        #         self.canvas_image_np,
        #         line_masks=self.line_masks,
        #         data_term_scale=self.data_scale.get(),
        #         smoothness_term_scale=self.smoothness_scale.get(),
        #     )
        self.graph_cut = GraphCut(
            self.canvas_image_np,
            rect=self.rectangle,
            line_masks=self.line_masks,
            data_term_scale=self.data_scale.get(),
            smoothness_term_scale=self.smoothness_scale.get(),
            apply_explicit_mask=self.apply_explicit_mask_var.get(),
        )
        self.mask = self.graph_cut.segment_2d()
        self.graph_cut.apply_explicit_mask = False

        self.img = cv.cvtColor(self.canvas_image_np, cv.COLOR_RGB2RGBA)
        self.img[:, :, 3] = self.mask * 255
        photo = ImageTk.PhotoImage(Image.fromarray(self.img))
        self.canvas_output_image = photo
        self.canvas_output.create_image(0, 0, anchor=tk.NW, image=photo)

        # self.grab_cut = GrabCut(self.canvas_image_np, self.rectangle)
        # image_np, mask = self.grab_cut.segment()
        # self.grab_cut = GrabCutOpenCV(self.canvas_image_np)
        # image_np, mask = self.grab_cut.segment(rect=self.rectangle)

        # photo = ImageTk.PhotoImage(Image.fromarray(image_np))
        # self.canvas_output_image = photo
        # self.canvas_output.create_image(0, 0, anchor=tk.NW, image=photo)
