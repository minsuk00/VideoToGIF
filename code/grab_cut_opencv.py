import numpy as np
import cv2 as cv
from typing import Optional


class GrabCutOpenCV:
    def __init__(self, image: np.ndarray):
        self.image = image  # (h x w x c)
        self.height, self.width = image.shape[:2]
        self.bg_model = np.zeros((1, 65), np.float64)
        self.fg_model = np.zeros((1, 65), np.float64)
        self.mask = None

    def segment(self, rect: Optional[list[int]] = None, lines: Optional[dict] = None):
        if rect:
            rect = [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]
            self.mask = np.zeros((self.height, self.width), np.uint8)
            self.mask, self.bg_model, self.fg_model = cv.grabCut(
                self.image, self.mask, rect, self.bg_model, self.fg_model, 5, cv.GC_INIT_WITH_RECT
            )
            mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype("uint8")
            img = self.image * mask2[:, :, np.newaxis]
            return img, mask2
        elif lines:
            self.mask[lines["fg"] == 1] = 1
            self.mask[lines["bg"] == 1] = 0
            self.mask, self.bg_model, self.fg_model = cv.grabCut(
                self.image, self.mask, None, self.bg_model, self.fg_model, 5, cv.GC_INIT_WITH_MASK
            )
            mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype("uint8")
            img = self.image * mask2[:, :, np.newaxis]
            return img, mask2

    def segment_frame_from_learnt_gmm(self, image: np.ndarray):
        # mask = np.zeros(image.shape[:2], np.uint8)
        # mask, _, _ = cv.grabCut(
        #     image, mask, [0, 0, self.width, self.height], self.bg_model, self.fg_model, 5, cv.GC_EVAL_FREEZE_MODEL
        # )
        mask = self.mask.copy()
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        cv.grabCut(image, mask, None, bg_model, fg_model, 2, cv.GC_INIT_WITH_MASK)
        # mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype("uint8")
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        print(np.any(mask != 0))
        img = cv.cvtColor(image, cv.COLOR_RGB2RGBA)
        img[:, :, 3] = mask2 * 255
        return img, mask2
