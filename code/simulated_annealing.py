import numpy as np
import random
import cv2 as cv
from sklearn.mixture import GaussianMixture
from scipy.optimize import dual_annealing


class SimulatedAnnealing:
    def __init__(self, image: np.ndarray, rect: list[int], temperature=1.0, cooling_rate=0.99, iterations=10):
        self.image = image  # (h x w x c)
        self.rect = rect
        self.height, self.width = image.shape[:2]
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.iterations = iterations

        self.n_components = 5
        self.mask = None
        self.fg_gmm = None
        self.bg_gmm = None

        self.init_mask(rect)
        self.init_gmms()

    def init_mask(self, rect):
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.mask[rect[1] : rect[3], rect[0] : rect[2]] = 1
        # cv.imshow("initial mask", self.mask * 255)
        # cv.waitKey(0)

    def init_gmms(self):
        """Fit GMMs to the initial mask."""
        fg_pixels = self.image[self.mask == 1].reshape(-1, 3)
        bg_pixels = self.image[self.mask == 0].reshape(-1, 3)

        # self.fg_gmm = GaussianMixture(n_components=self.n_components, covariance_type="full").fit(fg_pixels)
        # self.bg_gmm = GaussianMixture(n_components=self.n_components, covariance_type="full").fit(bg_pixels)
        self.fg_gmm = GaussianMixture(n_components=self.n_components).fit(fg_pixels)
        self.bg_gmm = GaussianMixture(n_components=self.n_components).fit(bg_pixels)

    def energy(self, segmentation):
        """Calculate the energy of the current segmentation."""
        smoothness_term = 0
        data_term = 0

        for i in range(segmentation.shape[0]):
            for j in range(segmentation.shape[1]):
                # Smoothness term
                if i > 0:
                    smoothness_term += segmentation[i, j] != segmentation[i - 1, j]
                if j > 0:
                    smoothness_term += segmentation[i, j] != segmentation[i, j - 1]

                # Data term
                pixel_value = self.image[i, j].reshape(1, 3)
                if segmentation[i, j] == 1:
                    data_term -= self.fg_gmm.score(pixel_value)  # TODO: precompute?
                else:
                    data_term -= self.bg_gmm.score(pixel_value)  # TODO: why minus?

        return smoothness_term + data_term

    def propose_change(self, segmentation):
        """Propose a change to the current segmentation."""
        new_segmentation = segmentation.copy()
        # i = random.randint(0, segmentation.shape[0] - 1)  # TODO: make it inside box
        # j = random.randint(0, segmentation.shape[1] - 1)
        i = random.randint(self.rect[1], self.rect[3])
        j = random.randint(self.rect[0], self.rect[2])
        new_segmentation[i, j] = 1 - new_segmentation[i, j]  # Flip the label (binary segmentation)
        return new_segmentation

    def accept_change(self, old_energy, new_energy):
        """Decide whether to accept the proposed change."""
        if new_energy < old_energy:
            return True
        else:
            probability = np.exp((old_energy - new_energy) / self.temperature)
            return random.random() < probability

    def run(self):
        """Run the simulated annealing algorithm."""
        current_segmentation = self.mask.copy()
        current_energy = self.energy(current_segmentation)

        # for _ in range(self.iterations):
        for _ in range(1):
            for i in range(self.height):
                for j in range(self.width):
                    new_segmentation = current_segmentation.copy()
                    new_segmentation[i, j] = 1 - new_segmentation[i, j]  # Flip the label (binary segmentation)

                    # new_segmentation = self.propose_change(current_segmentation)
                    new_energy = self.energy(new_segmentation)

                    if self.accept_change(current_energy, new_energy):
                        current_segmentation = new_segmentation
                        # current_energy = self.energy(current_segmentation)
                        current_energy = new_energy

                    self.temperature *= self.cooling_rate
                    # print("!")

        self.mask = current_segmentation
        return self.mask
