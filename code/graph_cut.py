import numpy as np
import cv2 as cv
from sklearn.mixture import GaussianMixture
import maxflow
from typing import Optional


class GraphCut:
    def __init__(
        self,
        image: np.ndarray,
        rect: Optional[list[int]] = None,
        line_masks: Optional[dict] = None,
        data_term_scale: float = 1.0,
        smoothness_term_scale: float = 1.0,
        apply_explicit_mask: bool = False,
    ):
        self.image = image  # (h x w x c)
        self.rect = rect
        self.line_masks = line_masks
        self.height, self.width = image.shape[:2]

        self.n_components = 5
        self.mask = None
        self.fg_gmm = None
        self.bg_gmm = None

        self.data_term_scale = data_term_scale
        self.smoothness_term_scale = smoothness_term_scale
        self.apply_explicit_mask = apply_explicit_mask

        self.init_mask()
        self.init_gmms()

    def init_mask(self):
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.rect:
            self.mask[self.rect[1] : self.rect[3], self.rect[0] : self.rect[2]] = 1
            self.mask = np.logical_not(self.mask).astype(np.uint8)
        if self.line_masks:
            self.mask[self.line_masks["bg"] == 1] = 1
            self.mask[self.line_masks["fg"] == 1] = 2
        # cv.imshow("initial mask", self.mask * 255)
        # cv.waitKey(0)

    def init_gmms(self):
        """Fit GMMs to the initial mask."""
        bg_pixels = self.image[self.mask == 1].reshape(-1, 3)
        fg_pixels = self.image[self.mask == 2].reshape(-1, 3)

        # self.fg_gmm = GaussianMixture(n_components=self.n_components, covariance_type="full").fit(fg_pixels)
        # self.bg_gmm = GaussianMixture(n_components=self.n_components, covariance_type="full").fit(bg_pixels)
        self.fg_gmm = GaussianMixture(n_components=self.n_components).fit(fg_pixels)
        self.bg_gmm = GaussianMixture(n_components=self.n_components).fit(bg_pixels)

    def calculate_edge_weight(self, pixel1, pixel2, gamma=0.01):
        return np.exp(-gamma * np.sum((pixel1 - pixel2) ** 2)) * self.smoothness_term_scale

    # ========================2D segmentation========================
    def build_graph_2d(self):
        # g = maxflow.Graph[int](self.height * self.width, self.height * self.width * 4)
        g = maxflow.Graph[float](self.height * self.width, self.height * self.width * 4)
        node_ids = g.add_nodes(self.height * self.width)

        fg_D = -self.fg_gmm.score_samples(self.image.reshape(-1, 3)) * self.data_term_scale
        bg_D = -self.bg_gmm.score_samples(self.image.reshape(-1, 3)) * self.data_term_scale
        # fg_D = np.where(fg_D < 0, 0, fg_D)
        # bg_D = np.where(bg_D < 0, 0, bg_D)

        cnt = 0
        for y in range(self.height):
            for x in range(self.width):
                index = y * self.width + x

                g.add_tedge(node_ids[index], bg_D[index], fg_D[index])

                if x < self.width - 1:
                    weight = self.calculate_edge_weight(self.image[y, x], self.image[y, x + 1])
                    g.add_edge(node_ids[index], node_ids[index + 1], weight, weight)
                    # g.add_edge(node_ids[index], node_ids[index + 1], 1, 1)
                if y < self.height - 1:
                    weight = self.calculate_edge_weight(self.image[y, x], self.image[y + 1, x])
                    g.add_edge(node_ids[index], node_ids[index + self.width], weight, weight)
                    # g.add_edge(node_ids[index], node_ids[index + self.width], 1, 1)

        return g, node_ids

    def segment_2d(self):
        g, node_ids = self.build_graph_2d()
        g.maxflow()
        segmentation = np.zeros((self.height, self.width), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                index = y * self.width + x
                if g.get_segment(node_ids[index]) == 0:
                    segmentation[y, x] = 1

        if self.apply_explicit_mask:
            segmentation[self.mask == 1] = 0
            segmentation[self.mask == 2] = 1
        return segmentation

    def segment_frame_from_learnt_gmm_2d(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.image = frame
        mask = self.segment_2d()

        img = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
        img[:, :, 3] = mask * 255
        return img, mask

    # ========================3D segmentation========================
    def build_graph_3d(self, prev_mask, energy_term_3d):
        g = maxflow.Graph[int](self.height * self.width, self.height * self.width * 4)
        # g = maxflow.Graph[float](self.height * self.width, self.height * self.width * 4)
        node_ids = g.add_nodes(self.height * self.width)

        fg_D = -self.fg_gmm.score_samples(self.image.reshape(-1, 3)) * self.data_term_scale
        bg_D = -self.bg_gmm.score_samples(self.image.reshape(-1, 3)) * self.data_term_scale

        for y in range(self.height):
            for x in range(self.width):
                index = y * self.width + x

                prev_bg_term = 0 if prev_mask[y, x] == 0 else energy_term_3d
                prev_fg_term = 0 if prev_mask[y, x] == 1 else energy_term_3d
                g.add_tedge(node_ids[index], bg_D[index] + prev_bg_term, fg_D[index] + prev_fg_term)

                if x < self.width - 1:
                    weight = self.calculate_edge_weight(self.image[y, x], self.image[y, x + 1])
                    g.add_edge(node_ids[index], node_ids[index + 1], weight, weight)
                    # g.add_edge(node_ids[index], node_ids[index + 1], 1, 1)
                if y < self.height - 1:
                    weight = self.calculate_edge_weight(self.image[y, x], self.image[y + 1, x])
                    g.add_edge(node_ids[index], node_ids[index + self.width], weight, weight)
                    # g.add_edge(node_ids[index], node_ids[index + self.width], 1, 1)

        return g, node_ids

    def segment_3d(self, prev_mask, energy_term_3d):
        g, node_ids = self.build_graph_3d(prev_mask, energy_term_3d)
        g.maxflow()
        segmentation = np.zeros((self.height, self.width), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                index = y * self.width + x
                if g.get_segment(node_ids[index]) == 0:
                    segmentation[y, x] = 1

        return segmentation

    def segment_frame_from_learnt_gmm_3d(
        self, frame: np.ndarray, prev_mask: np.ndarray, energy_term_3d
    ) -> tuple[np.ndarray, np.ndarray]:

        self.image = frame
        mask = self.segment_3d(prev_mask, energy_term_3d)

        img = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
        img[:, :, 3] = mask * 255
        return img, mask
