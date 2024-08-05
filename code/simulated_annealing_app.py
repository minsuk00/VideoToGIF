import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
from typing import Literal
import numpy as np

from code.grab_cut import GrabCut
from code.grab_cut_opencv import GrabCutOpenCV
from code.simulated_annealing import SimulatedAnnealing


class SimulatedAnnealingApp:
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
        self.current_phase: Literal["draw-rect", "process-image", "null"] = "null"
        self.start_x = None
        self.start_y = None
        self.current_item = None

        # Lists to keep track of the drawn shapes
        self.rectangle = None

    def update_mask(self, x, y):
        x0, y0, x1, y1 = x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size
        for i in range(max(0, y0), min(self.height, y1 + 1)):
            for j in range(max(0, x0), min(self.width, x1 + 1)):
                self.line_masks[self.mode][i][j] = 1

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.current_phase == "draw-rect":
            if self.current_item:
                self.canvas_input.delete(self.current_item)
            self.current_item = self.canvas_input.create_rectangle(
                self.start_x, self.start_y, event.x, event.y, outline="black"
            )

    def on_mouse_drag(self, event):
        if self.current_item:
            if self.current_phase == "draw-rect":
                self.canvas_input.coords(self.current_item, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.current_item:
            if self.current_phase == "draw-rect":
                self.rectangle = [int(val) for val in self.canvas_input.coords(self.current_item)]
                self.process_button.config(state=tk.NORMAL)
                self.current_phase = "process-image"
                print(self.canvas_input.coords(self.current_item))

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
            self.current_phase = "draw-rect"
            self.canvas_output.delete("all")
        else:
            print("Error. Failed to take snapshot from video.")

    def process_image(self):
        if self.current_phase == "process-image":
            self.mrf = SimulatedAnnealing(self.canvas_image_np, self.rectangle)

            mask = self.mrf.run()

            img = cv.cvtColor(self.canvas_image_np, cv.COLOR_RGB2RGBA)
            img[:, :, 3] = mask * 255
            photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas_output_image = photo
            self.canvas_output.create_image(0, 0, anchor=tk.NW, image=photo)

            # self.grab_cut = GrabCut(self.canvas_image_np, self.rectangle)
            # image_np, mask = self.grab_cut.segment()
            # self.grab_cut = GrabCutOpenCV(self.canvas_image_np)
            # image_np, mask = self.grab_cut.segment(rect=self.rectangle)

            # photo = ImageTk.PhotoImage(Image.fromarray(image_np))
            # self.canvas_output_image = photo
            # self.canvas_output.create_image(0, 0, anchor=tk.NW, image=photo)
