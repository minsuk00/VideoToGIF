import numpy as np
from sklearn.mixture import GaussianMixture
import networkx as nx
import cv2


import cProfile
import pstats


class GrabCut:
    def __init__(self, image: np.ndarray, rect: list[int]):
        self.image = image  # (h x w x c)
        self.rect = rect
        self.height, self.width = image.shape[:2]
        self.gamma = 50
        self.GMM_components = 5
        self.beta = self.calculate_beta()
        self.init_mask_from_rect()
        self.init_GMMs()

    # beta values for the smoothness term #TODO: get rid of?
    def calculate_beta(self):
        diffs = []
        for y in range(self.height):
            for x in range(self.width):
                if x > 0:
                    diffs.append(np.sum((self.image[y, x] - self.image[y, x - 1]) ** 2))
                if y > 0:
                    diffs.append(np.sum((self.image[y, x] - self.image[y - 1, x]) ** 2))
        return 1 / (2 * np.mean(diffs))

    def init_GMMs(self):
        self.background_gmm = self.learn_GMM_parameters(self.image[self.t_b == 1].reshape(-1, 3))
        self.foreground_gmm = self.learn_GMM_parameters(self.image[self.t_u == 1].reshape(-1, 3))
        # self.background_gmm = GaussianMixture(n_components=self.GMM_components)
        # self.foreground_gmm = GaussianMixture(n_components=self.GMM_components)

    def init_mask_from_rect(self):
        # create tu,tb,tf
        self.t_f = np.zeros((self.height, self.width), dtype=np.uint8)
        self.t_b = np.ones((self.height, self.width), dtype=np.uint8)
        self.t_u = np.zeros((self.height, self.width), dtype=np.uint8)

        self.t_b[self.rect[1] : self.rect[3], self.rect[0] : self.rect[2]] = 0  # Background region
        self.t_u[self.rect[1] : self.rect[3], self.rect[0] : self.rect[2]] = 1  # Unknown region

        self.alphas = np.zeros((self.height, self.width), dtype=np.uint8)
        self.alphas[self.rect[1] : self.rect[3], self.rect[0] : self.rect[2]] = 1

    def update_mask_from_lines(self, fgd_mask, bgd_mask):
        self.t_f = np.logical_or(self.t_f, fgd_mask).astype(np.uint8)
        self.t_b = np.logical_or(self.t_b, bgd_mask).astype(np.uint8)
        self.t_u = np.logical_not(np.logical_or(self.t_b, self.t_f)).astype(np.uint8)
        self.alphas = np.logical_or(self.t_u, self.t_f).astype(np.uint8)
        cv2.imshow("T_F", self.t_f * 255)
        cv2.imshow("T_B", self.t_b * 255)
        cv2.imshow("T_U", self.t_u * 255)
        cv2.imshow("Alphas", self.alphas * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def construct_graph(self):
        graph = nx.DiGraph()

        # fgd_D = -self.foreground_gmm.score_samples(self.image.reshape((-1, 3)))
        # bgd_D = -self.background_gmm.score_samples(self.image.reshape((-1, 3)))
        # TODO: precompute
        # TODO: only computer rectangle?
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                color = self.image[y, x]

                graph.add_edge("source", idx, capacity=-self.foreground_gmm.score([color]))
                graph.add_edge(idx, "sink", capacity=-self.background_gmm.score([color]))
                # graph.add_edge("source", idx, capacity=fgd_D[idx])
                # graph.add_edge(idx, "sink", capacity=bgd_D[idx])

                if x > 0:
                    left_idx = idx - 1
                    left_color = self.image[y, x - 1]
                    smoothness_cost = self.gamma * np.exp(-self.beta * np.sum((color - left_color) ** 2))
                    graph.add_edge(idx, left_idx, capacity=smoothness_cost)
                    graph.add_edge(left_idx, idx, capacity=smoothness_cost)

                if y > 0:
                    top_idx = idx - self.width
                    top_color = self.image[y - 1, x]
                    smoothness_cost = self.gamma * np.exp(-self.beta * np.sum((color - top_color) ** 2))
                    graph.add_edge(idx, top_idx, capacity=smoothness_cost)
                    graph.add_edge(top_idx, idx, capacity=smoothness_cost)

        return graph

    def min_cut(self, graph):
        cut_value, (reachable, non_reachable) = nx.minimum_cut(graph, "source", "sink")

        return reachable

    def update_mask(self, visited):
        mask = np.zeros((self.height, self.width), np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                if idx in visited:
                    mask[y, x] = 1
        mask[self.t_f == 1] = 1
        mask[self.t_b == 1] = 0
        return mask

    def learn_GMM_parameters(self, pixels):
        gmm = GaussianMixture(n_components=self.GMM_components)
        gmm.fit(pixels)
        return gmm

    def assign_GMM_components(self, pixels):
        fgd_log_prob = self.foreground_gmm.score_samples(pixels)
        bgd_log_prob = self.background_gmm.score_samples(pixels)

        fgd_pixels = pixels[fgd_log_prob > bgd_log_prob]
        bgd_pixels = pixels[fgd_log_prob <= bgd_log_prob]
        return fgd_pixels, bgd_pixels

    def iterate(self):
        self.init_GMMs()
        # Iterate multiple times for convergence
        for _ in range(5):
            # Step 1
            bgd_pixels = self.image[self.t_b == 1].reshape(-1, 3)
            # print(bgd_pixels.shape)  # (86805, 3)
            fgd_pixels = self.image[self.t_f == 1].reshape(-1, 3)
            unknown_pixels = self.image[self.t_u == 1].reshape(-1, 3)

            bgd_predicted_pixels, fgd_predicted_pixels = self.assign_GMM_components(unknown_pixels)
            fgd_pixels = np.append(fgd_pixels, fgd_predicted_pixels, axis=0)
            bgd_pixels = np.append(bgd_pixels, bgd_predicted_pixels, axis=0)

            # Step 2
            if len(bgd_pixels) > 0:
                self.background_gmm = self.learn_GMM_parameters(bgd_pixels)
            if len(fgd_pixels) > 0:
                self.foreground_gmm = self.learn_GMM_parameters(fgd_pixels)

            # Step 3
            print("Step 3 begin")
            graph = self.construct_graph()
            print("graph construction finished")
            visited = self.min_cut(graph)
            print("min cut finished")
            self.alphas = self.update_mask(visited)
            break

    def segment(self):
        profiler = cProfile.Profile()
        profiler.enable()
        self.iterate()
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumulative").print_stats(10)

        output_image = self.image.copy()
        output_image = np.dstack(
            (output_image, np.ones((output_image.shape[0], output_image.shape[1]), dtype=np.uint8) * 255)
        )
        output_image[self.alphas == 0, 3] = 0

        return output_image, self.alphas
