import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv
from typing import Literal
import numpy as np
import os


class VideoSegmentationApp:
    def __init__(self, root, graph_cut_app, video_player):
        self.root = root
        self.video_player = video_player
        self.graph_cut_app = graph_cut_app
        self.width = graph_cut_app.width
        self.height = graph_cut_app.height

        # ===================== BUTTONS =====================
        button_frame = tk.Frame(root)
        button_frame.pack()

        # Run button
        self.run_button = tk.Button(button_frame, text="Run Video Segmentation", command=self.run)
        self.run_button.grid(row=0, column=0)

        # 3D vs 2D toggle button
        self.toggle_var_3d = tk.BooleanVar(value=False)
        self.toggle_button_3d = tk.Checkbutton(
            button_frame, text="3D", variable=self.toggle_var_3d, command=self.toggle_3d
        )
        self.toggle_button_3d.grid(row=0, column=1)

        # Slider for 3D term
        scale_frame = tk.Frame(button_frame)
        scale_frame.grid(row=0, column=2)
        self.slider_3d_term = tk.Scale(scale_frame, from_=0, to=10, orient="horizontal", name="hi")
        self.slider_3d_term.set(3)  # Set initial value
        self.slider_3d_term.pack()
        scale_label = tk.Label(scale_frame, text="3D term")
        scale_label.pack(side=tk.BOTTOM)
        # ===================================================

        # ===================== LABEL =======================
        # self.canvas_output = tk.Canvas(
        #     root,
        #     width=self.width,
        #     height=self.height,
        #     bg="black",
        #     highlightthickness=0,
        # )
        # self.canvas_output.pack(pady=10)
        gif_frame = tk.Frame(root, width=self.width, height=self.height)
        gif_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents
        gif_frame.pack(pady=10)
        self.label = tk.Label(
            gif_frame,
            background="black",
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        self.play_button = tk.Button(
            root, text="Play GIF", command=lambda: self.show_result_to_canvas(0), state=tk.DISABLED
        )
        self.play_button.pack(pady=10)

        # Download button
        self.download_button = tk.Button(root, text="Download GIF", command=self.download, state=tk.DISABLED)
        self.download_button.pack()
        # ===================================================

        self.output_frames = []
        self.fps = 30
        self.delay = 1000 / self.fps

        self.is_3d: bool = True
        self.initial_frame_num = None

    def toggle_3d(self):
        if self.toggle_var_3d.get():
            self.toggle_button_3d.config(text="3D")
            self.is_3d = True
        else:
            self.toggle_button_3d.config(text="2D")
            self.is_3d = False

    def run(self):
        if self.graph_cut_app.graph_cut is None:
            print("run graph cut first")
            return
        # if self.initial_frame_num is None:
        #     self.initial_frame_num = self.video_player.current_frame + 1
        initial_frame_num = self.video_player.current_frame + 1
        prev_mask = self.graph_cut_app.mask
        self.output_frames = []
        self.output_frames.append(self.graph_cut_app.img)

        self.video_player.cap.set(cv.CAP_PROP_POS_FRAMES, initial_frame_num)
        while 1:
            _, np_frame = self.video_player.capture_current_frame()
            if np_frame is None:
                break
            if self.is_3d:
                img, mask = self.graph_cut_app.graph_cut.segment_frame_from_learnt_gmm_3d(
                    np_frame, prev_mask, self.slider_3d_term.get()
                )
                prev_mask = mask
            else:
                img, mask = self.graph_cut_app.graph_cut.segment_frame_from_learnt_gmm_2d(np_frame)
            self.output_frames.append(img)
        self.download_button.config(state=tk.NORMAL)
        self.play_button.config(state=tk.NORMAL)
        self.output_frames = [Image.fromarray(frame) for frame in self.output_frames]
        print("num of frames in output GIF:", len(self.output_frames))
        self.show_result_to_canvas(0)
        self.video_player.cap.set(cv.CAP_PROP_POS_FRAMES, initial_frame_num)

    def show_result_to_canvas(self, frame_idx: int):
        frame = self.output_frames[frame_idx]
        photo = ImageTk.PhotoImage(frame)
        self.label.config(image=photo)
        self.label.image = photo
        # next_idx = (frame_idx + 1) % len(self.output_frames)
        if frame_idx + 1 < len(self.output_frames):
            self.root.after(int(self.delay), self.show_result_to_canvas, frame_idx + 1)

    def download(self):
        # Save as a GIF
        gif_name = "output.gif"
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        gif_path = os.path.join(downloads_dir, gif_name)

        self.output_frames[0].save(
            gif_path,
            save_all=True,
            append_images=self.output_frames[1:],
            loop=0,
            duration=self.delay,  # TODO: calculate delay?
            transparency=0,
            disposal=2,
        )
        print(f"GIF saved to {gif_path}!")
