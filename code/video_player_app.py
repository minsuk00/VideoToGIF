import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk


class VideoPlayerApp:
    def __init__(self, root, video_width=640, video_height=480):
        self.root = root

        self.fps = 30
        self.video_width = video_width
        self.video_height = video_height

        self.cap = None
        self.paused = True

        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack(pady=10)

        self.label = tk.Label(root)
        self.label.pack()

        # Create a frame for the button and slider
        control_frame = tk.Frame(root)
        # control_frame.pack(fill=tk.X, padx=10, pady=10)
        control_frame.pack(fill=tk.X, padx=10)

        self.play_pause_button = tk.Button(
            control_frame, width=6, height=1, text="Play", command=self.play_pause_video, state=tk.DISABLED
        )
        self.play_pause_button.grid(row=0, column=0, padx=10)

        self.slider = tk.Scale(
            control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_video, state=tk.DISABLED
        )
        self.slider.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(0, 16))
        control_frame.columnconfigure(1, weight=1)  # Make the second column expandable

        self.total_frames = 0
        self.current_frame = 0

        self.video_path = "./assets/totoro.mp4"
        self.load_video()

    def load_video(self):
        # self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.slider.config(to=self.total_frames)
            self.play_pause_button.config(state=tk.NORMAL)
            self.slider.config(state=tk.NORMAL)
            self.reset_video()

            num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Video loaded: {num_frames} frames, {num_frames/fps} seconds")

            # show initial frame
            photo, _ = self.capture_current_frame()
            if photo:
                self.label.config(image=photo)
                self.label.image = photo
        else:
            print("No video path available")

    def play_pause_video(self):
        if self.cap is None:
            print("No video capture")
            return

        self.paused = not self.paused
        self.play_pause_button.config(text="Play" if self.paused else "Pause")

        if not self.paused:
            self.play_video()

    def play_video(self):
        if self.cap is None or self.paused:
            return

        photo, _ = self.capture_current_frame()
        if photo:
            self.label.config(image=photo)
            self.label.image = photo

            self.current_frame += 1
            self.slider.set(self.current_frame)

            # Calculate delay based on fps
            delay = int(1000 / self.fps)
            self.root.after(delay, self.play_video)
        else:
            self.reset_video()

    def reset_video(self):
        self.paused = True
        self.play_pause_button.config(text="Play")
        self.slider.set(0)
        self.seek_video(0)

    def seek_video(self, value):
        if self.cap is None:
            return
        self.current_frame = int(value)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        # change frame to new location
        photo, _ = self.capture_current_frame()
        if photo:
            self.label.config(image=photo)
            self.label.image = photo
        # self.play_video()

    def capture_current_frame(self):
        ret, frame = self.cap.read()
        photo = None
        if ret:
            frame = cv2.resize(frame, (self.video_width, self.video_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
        return photo, frame
