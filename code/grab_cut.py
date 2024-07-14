import numpy as np
from sklearn.mixture import GaussianMixture


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

        self.segment()

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

    def construct_graph(self):
        edges = []
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                color = self.image[y, x]

                if self.alphas[y, x] == 0:  # Background
                    edges.append(("source", idx, -self.background_gmm.score([color])))
                    edges.append((idx, "sink", -self.foreground_gmm.score([color])))
                elif self.alphas[y, x] == 1:  # Foreground
                    edges.append(("source", idx, -self.background_gmm.score([color])))
                    edges.append((idx, "sink", -self.foreground_gmm.score([color])))

                if x > 0:
                    left_idx = idx - 1
                    left_color = self.image[y, x - 1]
                    smoothness_cost = self.gamma * np.exp(-self.beta * np.sum((color - left_color) ** 2))
                    edges.append((idx, left_idx, smoothness_cost))
                    edges.append((left_idx, idx, smoothness_cost))

                if y > 0:
                    top_idx = idx - self.width
                    top_color = self.image[y - 1, x]
                    smoothness_cost = self.gamma * np.exp(-self.beta * np.sum((color - top_color) ** 2))
                    edges.append((idx, top_idx, smoothness_cost))
                    edges.append((top_idx, idx, smoothness_cost))

        return edges

    def min_cut(self, edges):
        capacity = {}
        flow = {}
        for u, v, cap in edges:
            if u != v:
                capacity[(u, v)] = cap
                capacity[(v, u)] = 0
                flow[(u, v)] = 0
                flow[(v, u)] = 0

        height = {}
        excess = {}
        vertices = set()
        for u, v, cap in edges:
            vertices.add(u)
            vertices.add(v)

        for v in vertices:
            height[v] = 0
            excess[v] = 0
        height["source"] = len(vertices)

        for u, v in capacity:
            if u == "source":
                flow[(u, v)] = capacity[(u, v)]
                excess[v] = capacity[(u, v)]
                excess["source"] -= capacity[(u, v)]

        def push(u, v):
            amt = min(excess[u], capacity[(u, v)] - flow[(u, v)])
            flow[(u, v)] += amt
            flow[(v, u)] -= amt
            excess[u] -= amt
            excess[v] += amt

        def relabel(u):
            min_height = float("inf")
            for v in vertices:
                if (u, v) in capacity and capacity[(u, v)] > flow[(u, v)]:
                    min_height = min(min_height, height[v])
            height[u] = min_height + 1

        def discharge(u):
            while excess[u] > 0:
                for v in vertices:
                    if (u, v) in capacity and capacity[(u, v)] > flow[(u, v)]:
                        if height[u] > height[v]:
                            push(u, v)
                            if excess[u] == 0:
                                break
                if excess[u] > 0:
                    relabel(u)

        active = [v for v in vertices if v != "source" and v != "sink" and excess[v] > 0]
        while active:
            u = active.pop(0)
            discharge(u)
            if excess[u] > 0:
                active.append(u)

        visited = set()

        def bfs(source):
            queue = [source]
            while queue:
                u = queue.pop(0)
                for v in capacity:
                    if (u, v) in flow and flow[(u, v)] < capacity[(u, v)] and v not in visited:
                        visited.add(v)
                        queue.append(v)

        bfs("source")

        return visited

    def update_mask(self, visited):
        mask = np.zeros((self.height, self.width), np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                idx = y * self.width + x
                if idx in visited:
                    mask[y, x] = 1
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
            edges = self.construct_graph()
            print("graph construction finished")
            visited = self.min_cut(edges)
            print("min cut finished")
            self.alphas = self.update_mask(visited)
            break

    def segment(self):
        self.iterate()
        return self.alphas
