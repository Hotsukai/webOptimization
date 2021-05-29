import numpy as np
import colorsys
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def hl_to_rgb(x):
    return colorsys.hls_to_rgb(x[0], x[1], 1)


N = 30
X_im = np.zeros((N, N, 2))
rs = []
xs = []
for i in range(N):
    for j in range(N):
        X_im[i, j, 0] = i / N  # Hue
        X_im[i, j, 1] = j / N  # Lightness
X_rgb = np.apply_along_axis(hl_to_rgb, -1, X_im)


X_star = X_im.reshape((N * N, 2)).T


class GPUCBAgent(object):
    def __init__(self):
        self.xs = []
        self.rs = []
        self.gamma = 10
        self.s = 0.5
        self.alpha = 2
        self.Q_starstar = X_star.T.dot(X_star)
        self.K_starstar = np.exp(
            -self.gamma
            * (
                np.tile(np.diag(self.Q_starstar), (X_star.shape[1], 1)).T
                - 2 * self.Q_starstar
                + np.tile(np.diag(self.Q_starstar), (X_star.shape[1], 1))
            )
        )
        self.mu = np.zeros(self.K_starstar.shape[0])
        self.sigma = self.K_starstar

    def get_arm(self):
        ucb = self.mu + self.alpha * np.diag(self.sigma)
        return X_star[:, np.argmax(ucb)], ucb

    def sample(self, x, r):
        self.xs.append(x)
        self.rs.append(r)
        X = np.array(self.xs).T

        Q = X.T.dot(X)
        Q_star = X.T.dot(X_star)
        K = np.exp(
            -self.gamma
            * (
                np.tile(np.diag(Q), (X.shape[1], 1)).T
                - 2 * Q
                + np.tile(np.diag(Q), (X.shape[1], 1))
            )
        )
        K_star = np.exp(
            -self.gamma
            * (
                np.tile(np.diag(Q), (X_star.shape[1], 1)).T
                - 2 * Q_star
                + np.tile(np.diag(self.Q_starstar), (X.shape[1], 1))
            )
        )
        A = np.linalg.inv(self.s + np.identity(K.shape[0]) + K)
        self.mu = K_star.T.dot(A).dot(self.rs)
        self.sigma = self.K_starstar - K_star.T.dot(A).dot(K_star)


class GPTSAgent(object):
    def __init__(self):
        self.xs = []
        self.rs = []
        self.gamma = 10
        self.s = 0.5
        self.Q_starstar = X_star.T.dot(X_star)
        self.K_starstar = np.exp(
            -self.gamma
            * (
                np.tile(np.diag(self.Q_starstar), (X_star.shape[1], 1)).T
                - 2 * self.Q_starstar
                + np.tile(np.diag(self.Q_starstar), (X_star.shape[1], 1))
            )
        )
        self.mu = np.zeros(self.K_starstar.shape[0])
        self.sigma = self.K_starstar

    def get_arm(self):
        f = np.random.multivariate_normal(self.mu, self.sigma)
        return X_star[:, np.argmax(f)], f

    def sample(self, x, r):
        self.xs.append(x)
        self.rs.append(r)
        X = np.array(self.xs).T

        Q = X.T.dot(X)
        Q_star = X.T.dot(X_star)
        K = np.exp(
            -self.gamma
            * (
                np.tile(np.diag(Q), (X.shape[1], 1)).T
                - 2 * Q
                + np.tile(np.diag(Q), (X.shape[1], 1))
            )
        )
        K_star = np.exp(
            -self.gamma
            * (
                np.tile(np.diag(Q), (X_star.shape[1], 1)).T
                - 2 * Q_star
                + np.tile(np.diag(self.Q_starstar), (X.shape[1], 1))
            )
        )
        A = np.linalg.inv(self.s + np.identity(K.shape[0]) + K)
        self.mu = K_star.T.dot(A).dot(self.rs)
        self.sigma = self.K_starstar - K_star.T.dot(A).dot(K_star)


def getPlt(agent, x, f):
    vmax = 1.6
    vmin = -1.6
    contour_linewidth = 0.6
    contour_fontsize = 6
    contour_levels = np.linspace(-2, 2, 17)
    fig = plt.figure()
    grid = ImageGrid(fig, 211, nrows_ncols=(1, 2), axes_pad=0.1)
    grid[0].imshow(X_rgb)
    cs = grid[0].contour(
        f.reshape(N, N),
        levels=contour_levels,
        colors="white",
        linewidths=contour_linewidth,
    )
    grid[0].clabel(cs, inline=1, fontsize=contour_fontsize)
    grid[0].plot(
        x[1] * N,
        x[0] * N,
        "*",
        markersize=20,
        color="yellow",
        markeredgecolor="black",
    )
    grid[0].set_title("Solution space")
    grid[0].set_xticklabels([])
    grid[0].set_yticklabels([])
    grid[1].imshow(np.tile(hl_to_rgb(x), (N, N, 1)))
    grid[1].set_title("Proposed color")
    grid[1].set_xticklabels([])
    grid[1].set_yticklabels([])
    grid = ImageGrid(
        fig,
        212,
        nrows_ncols=(1, 3),
        axes_pad=0.2,
        share_all=True,
        label_mode="L",
        cbar_location="left",
        cbar_mode="single",
    )
    im = grid[0].imshow(agent.mu.reshape(N, N), vmin=vmin, vmax=vmax)
    cs = grid[0].contour(
        agent.mu.reshape(N, N),
        levels=contour_levels,
        colors="white",
        linewidths=contour_linewidth,
    )
    grid[0].clabel(cs, inline=1, fontsize=contour_fontsize)
    grid[0].set_title(r"$\mu$")
    grid.cbar_axes[0].colorbar(im)
    grid[1].imshow(np.diag(agent.sigma).reshape(N, N), vmin=vmin, vmax=vmax)
    cs = grid[1].contour(
        np.diag(agent.sigma).reshape(N, N),
        levels=contour_levels,
        colors="white",
        linewidths=contour_linewidth,
    )
    grid[1].set_title(r"$diag(\Sigma)$")
    grid[1].clabel(cs, inline=1, fontsize=contour_fontsize)
    grid[2].imshow(f.reshape(N, N), vmin=vmin, vmax=vmax)
    cs = grid[2].contour(
        f.reshape(N, N),
        levels=contour_levels,
        colors="black",
        linewidths=contour_linewidth,
    )
    grid[2].clabel(cs, inline=1, fontsize=contour_fontsize)
    grid[2].set_title("Acquisition function")
    # plt.show()
    return plt
