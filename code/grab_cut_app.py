import tkinter as tk
import io
from PIL import Image, ImageTk
import cv2
from typing import Literal

from code.grab_cut import GrabCut


class GrabCutApp:
    def __init__(self, root, video_player):
        self.root = root
        self.video_player = video_player

        # Take snapshot button
        self.snapshot_button = tk.Button(root, text="Take Snapshot", command=self.take_snapshot)
        self.snapshot_button.pack(side=tk.LEFT, pady=10)

        # Process button
        self.process_button = tk.Button(root, text="Process Image", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(side="left", pady=10)

        # Canvas for displaying snapshot and drawing
        self.canvas = tk.Canvas(
            root,
            width=self.video_player.video_width,
            height=self.video_player.video_height,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)

        self.canvas_image = None
        self.canvas_image_id = None
        self.canvas_image_np = None

        # Bind mouse events to canvas
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        # ========================================

        self.current_phase: Literal["draw-rect", "user-edit"] = "draw-rect"
        self.start_x = None
        self.start_y = None
        self.current_item = None

        # Lists to keep track of the drawn shapes
        self.rectangle = None
        self.lines = {"fg": [], "bg": []}

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.current_phase == "draw-rect":
            if self.current_item:
                self.canvas.delete(self.current_item)
            self.current_item = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y, outline="black"
            )
        elif self.current_phase == "user-edit":
            self.current_item = self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill="black")

    def on_mouse_drag(self, event):
        if self.current_item:
            if self.current_phase == "draw-rect":
                self.canvas.coords(self.current_item, self.start_x, self.start_y, event.x, event.y)
            elif self.current_phase == "user-edit":
                self.canvas.coords(self.current_item, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.current_item:
            if self.current_phase == "draw-rect":
                self.rectangle = [int(val) for val in self.canvas.coords(self.current_item)]
                self.process_button.config(state=tk.NORMAL)
                print(self.canvas.coords(self.current_item))
            elif self.current_phase == "user-edit":
                self.lines.append(self.canvas.coords(self.current_item))
            # self.current_item = None

    def take_snapshot(self):
        photo, np_photo = self.video_player.capture_current_frame()
        if photo:
            # keep reference in order to prevent gc
            self.canvas_image = photo
            self.canvas_image_np = np_photo  # (h x w x c)

            # if self.canvas_image_id is not None:
            #     self.canvas.delete(self.canvas_image_id)
            # self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            # self.canvas.create_image(0, 0, image=photo)
            self.process_button.config(state=tk.DISABLED)
            self.current_phase = "draw-rect"
        else:
            print("Error. Failed to take snapshot from video.")

    def process_image(self):
        grab_cut = GrabCut(self.canvas_image_np, self.rectangle)

        self.current_phase = "user-edit"
