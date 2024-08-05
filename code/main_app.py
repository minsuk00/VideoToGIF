import io
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Canvas
import cv2
from PIL import Image, ImageTk
import numpy as np

from code.video_player_app import VideoPlayerApp
from code.grab_cut_app import GrabCutApp
from code.simulated_annealing_app import SimulatedAnnealingApp
from code.video_segmentation_app import VideoSegmentationApp
from code.graph_cut_app import GraphCutApp


class VideoToGIF:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video To GIF")
        self.root.geometry("1000x1200-200-300")
        # self.root.resizable(True, True)

        # set icon
        icon_path = "./assets/profile.png"
        icon_image = Image.open(icon_path)
        icon_photo = ImageTk.PhotoImage(icon_image)
        root.iconphoto(False, icon_photo)

        VIDEO_WIDTH = 426
        VIDEO_HEIGHT = 240
        # VIDEO_WIDTH = 177
        # VIDEO_HEIGHT = 100
        self.video_player = VideoPlayerApp(root, VIDEO_WIDTH, VIDEO_HEIGHT)
        # self.grab_cut_app = GrabCutApp(root, self.video_player)
        # self.grab_cut_app = SimulatedAnnealingApp(root, self.video_player)
        self.graph_cut_app = GraphCutApp(root, self.video_player)
        self.video_segmentation_app = VideoSegmentationApp(root, self.graph_cut_app, self.video_player)

        # Set focus to the main window and update
        self.root.update_idletasks()
        self.root.focus_force()
        # self.root.update()
