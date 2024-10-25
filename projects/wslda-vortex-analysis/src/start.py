#!/usr/bin/env python3

from enum import Enum
from matplotlib import ticker
from scipy.interpolate import interp1d
from typing import Any, Union, Tuple, List, Set, Dict, Callable
from wdata.io import WData, Var
import matplotlib
import matplotlib.axes
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import pprint
import sys


# helpers:


class Axis(int, Enum):
    X = 0
    Y = 1
    Z = 2


class Component(int, Enum):
    X = 0
    Y = 1
    Z = 2


class ParsedWData(object):
    def __init__(
        self,
        wdata: WData,
        iteration: int
    ):
        # data:
        # - grid:
        self.Nx: int = wdata.Nxyz[Axis.X]
        self.Ny: int = wdata.Nxyz[Axis.Y]
        self.dx: float = wdata.dxyz[Axis.X]
        self.dy: float = wdata.dxyz[Axis.Y]
        self.x: numpy.ndarray = wdata.xyz[Axis.X]
        self.y: numpy.ndarray = wdata.xyz[Axis.Y]
        self.x_flat: List[float] = self.x.flatten()
        self.y_flat: List[float] = self.y.flatten()

        # data:
        # - current density
        self.j_a_x: numpy.memmap = wdata.j_a[iteration][Component.X]
        self.j_a_y: numpy.memmap = wdata.j_a[iteration][Component.Y]
        self.j_a: List[List[float]] = [[j_a_x, j_a_y] for j_a_x, j_a_y in zip(self.j_a_x, self.j_a_y)]
        self.j_b_x: numpy.memmap = wdata.j_b[iteration][Component.X]
        self.j_b_y: numpy.memmap = wdata.j_b[iteration][Component.Y]
        self.j_b: List[List[float]] = [[j_b_x, j_b_y] for j_b_x, j_b_y in zip(self.j_b_x, self.j_b_y)]


class Plot():
    def __init__(
        self,
        rows_h_ratios,
        cols_w_ratios,
        subplot_h,
        subplot_w,
    ):
        # plot's width and height:
        self.subplot_h: float = subplot_h
        self.subplot_w: float = subplot_w
        self.nrows: float = len(rows_h_ratios)
        self.ncols: float = len(cols_w_ratios)
        self.rows_h_ratios: List[float] = rows_h_ratios
        self.cols_w_ratios: List[float] = cols_w_ratios
        self.h: float = np.sum([self.subplot_h * ratio for ratio in (self.rows_h_ratios / np.max(self.rows_h_ratios))])
        self.w: float = np.sum([self.subplot_w * ratio for ratio in (self.cols_w_ratios / np.max(self.cols_w_ratios))])

        # create plot and subplots:
        self.fig = plt.figure(
            figsize = (self.w, self.h)
        )
        self.gs = gridspec.GridSpec(
            nrows = self.nrows,
            ncols = self.ncols,
            height_ratios = self.rows_h_ratios,
            width_ratios  = self.cols_w_ratios,
        )
        self.ax: List[List[matplotlib.axes.Axes]] = [[None for _ in range(self.ncols)] for _ in range(self.nrows)]
        for i in range(0, self.nrows):
            for j in range(0, self.ncols):
                self.ax[i][j] = self.fig.add_subplot(self.gs[i, j])


class CustomScientificFormatter(ticker.ScalarFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x, pos=None):
        if x == 0:
            return "0"  # Handle zero separately
        else:
            return f"{x:.1e}"  # Format to scientific notation, e.g., 1.0e-3


def gen_scientific_formatter() -> ticker.Formatter:
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2)) # Use scientific notation if <1e-2 or >1e2

    return formatter


# partial plots


def plot_streamplot(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    x: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    data_x: numpy.typing.ArrayLike,
    data_y: numpy.typing.ArrayLike,
    title,
    label_x,
    label_y,
) -> None:
    # Plot data:
    ax.streamplot(x, y, data_x, data_y, color='blue', linewidth=2, density=1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    # Set axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))


def plot_pseudocolor(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    x: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    data: numpy.typing.ArrayLike,
    title,
    label_x,
    label_y,
) -> None:
    # Plot data:
    pcolor = ax.pcolormesh(x, y, data, cmap='plasma', shading='auto')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    # Set axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    # Add color bar
    color_bar = fig.colorbar(
        pcolor,
        ax = ax,
        location = 'right',
        orientation = 'vertical',
        fraction = 0.035,
        pad = 0.01,
        # location='left',
        # orientation='vertical',
        # fraction=0.035,
        # pad=-0.16
    )

    # Set scientific notation for color bar labels
    formatter = gen_scientific_formatter()
    color_bar.ax.yaxis.set_major_formatter(formatter)


def plot_crossections_of_2D_data(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    indices: List[int],
    x: numpy.typing.ArrayLike,
    data: numpy.typing.ArrayLike,
    gen_csec_func: Callable[[numpy.typing.ArrayLike, int], numpy.typing.ArrayLike],
    title,
    label_x,
    label_y,
    gen_label_data_func: Callable[[numpy.typing.ArrayLike, int], str],
) -> None:
    # The default color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot data:
    for i, color in zip(indices, colors):
        data_csec = gen_csec_func(data, i)

        int_x = np.linspace(x[0], x[-1], 200)
        int_data = interp1d(x, data_csec, kind='quadratic')

        # plot points:
        ax.plot(
            x,
            data_csec,
            label=gen_label_data_func(x, i),
            marker='o',
            markersize=4,
            linestyle='None',
            color=color
        )

        # plot interpolation:
        ax.plot(
            int_x,
            int_data(int_x),
            marker='None',
            linestyle="-",
            color=color
        )

    # Labels
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    # Axis limits
    ax.set_xlim(1.05 * np.min(x), 1.05 * np.max(x))
    ax.set_ylim(1.05 * np.min(data), 1.05 * np.max(data))

    # Format axis labels to scientific notation
    formatter = gen_scientific_formatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add legend
    ax.legend()


# plot systen properties:


def plot_current_density(
    wdata: WData,
    iteration: int
):
    # parse WData:
    d = ParsedWData(wdata, iteration)

    # gen plot:
    p = Plot(
        cols_w_ratios = [1, 1, 1, 1, 1, 1, 1],
        rows_h_ratios = [1, 1],
        subplot_w = 10,
        subplot_h = 10,
    )

    # plot:
    # - WARNING: for matplotlib to display data, p.j_*_* needs to be transposed !!!
    #     according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
    #     - wdata.j_a[iteration][Component.X][ Nx/2, :    ] === j_a_x( x = Nx/2, y        )
    #     - wdata.j_a[iteration][Component.X][ :   , Ny/2 ] === j_a_x( x       , y = Ny/2 )
    nx_half = int(d.Nx/2)
    ny_half = int(d.Ny/2)
    x_crossections_01 = [nx_half - 10, nx_half - 5, nx_half - 2, nx_half]
    x_crossections_02 = [nx_half - 10, nx_half - 5, nx_half - 2, nx_half, nx_half + 2, nx_half + 5, nx_half + 10]
    y_crossections_01 = [ny_half - 10, ny_half - 5, ny_half - 2, ny_half]
    y_crossections_02 = [ny_half - 10, ny_half - 5, ny_half - 2, ny_half, ny_half + 2, ny_half + 5, ny_half + 10]

    plot_streamplot(p.fig, p.ax[0][0], d.x_flat, d.y_flat, d.j_a_x.T, d.j_a_y.T, 'j_a', 'x', 'y')
    plot_streamplot(p.fig, p.ax[1][0], d.x_flat, d.y_flat, d.j_b_x.T, d.j_b_y.T, 'j_b', 'x', 'y')

    plot_pseudocolor(p.fig, p.ax[0][1], d.x_flat, d.y_flat, d.j_a_x.T, 'j_a_x', 'x', 'y')
    plot_pseudocolor(p.fig, p.ax[0][4], d.x_flat, d.y_flat, d.j_a_y.T, 'j_a_y', 'x', 'y')
    plot_pseudocolor(p.fig, p.ax[1][1], d.x_flat, d.y_flat, d.j_b_x.T, 'j_b_x', 'x', 'y')
    plot_pseudocolor(p.fig, p.ax[1][4], d.x_flat, d.y_flat, d.j_b_y.T, 'j_b_y', 'x', 'y')

    plot_crossections_of_2D_data(p.fig, p.ax[0][2], x_crossections_01, d.y_flat, d.j_a_x, lambda data, i: data[i, :], 'j_a_x', 'y', 'j_a_x', lambda x, i: f"j_a_x(x = {x[i]}, y)")
    plot_crossections_of_2D_data(p.fig, p.ax[0][3], y_crossections_02, d.x_flat, d.j_a_x, lambda data, i: data[:, i], 'j_a_x', 'x', 'j_a_x', lambda y, i: f"j_a_x(x, y = {y[i]})")

    plot_crossections_of_2D_data(p.fig, p.ax[0][5], x_crossections_02, d.y_flat, d.j_a_y, lambda data, i: data[i, :], 'j_a_y', 'y', 'j_a_y', lambda x, i: f"j_a_y(x = {x[i]}, y)")
    plot_crossections_of_2D_data(p.fig, p.ax[0][6], y_crossections_01, d.x_flat, d.j_a_y, lambda data, i: data[:, i], 'j_a_y', 'x', 'j_a_y', lambda y, i: f"j_a_y(x, y = {y[i]})")

    plot_crossections_of_2D_data(p.fig, p.ax[1][2], x_crossections_01, d.y_flat, d.j_b_x, lambda data, i: data[i, :], 'j_b_x', 'y', 'j_b_x', lambda x, i: f"j_b_x(x = {x[i]}, y)")
    plot_crossections_of_2D_data(p.fig, p.ax[1][3], y_crossections_02, d.x_flat, d.j_b_x, lambda data, i: data[:, i], 'j_b_x', 'x', 'j_b_x', lambda y, i: f"j_b_x(x, y = {y[i]})")

    plot_crossections_of_2D_data(p.fig, p.ax[1][5], x_crossections_02, d.y_flat, d.j_b_y, lambda data, i: data[i, :], 'j_b_y', 'y', 'j_b_y', lambda x, i: f"j_b_y(x = {x[i]}, y)")
    plot_crossections_of_2D_data(p.fig, p.ax[1][6], y_crossections_01, d.x_flat, d.j_b_y, lambda data, i: data[:, i], 'j_b_y', 'x', 'j_b_y', lambda y, i: f"j_b_y(x, y = {y[i]})")

    # save plot:
    p.fig.tight_layout()
    p.fig.savefig(
        fname = f"plot_current_density.png",
        dpi = 300
    )


# start:


def main() -> int:
    """
    entry point
    """

    inDir = "/workspace/results/data"
    outDir = "/workspace/results/analysis"
    inFile = inDir + "/st-vortex-recreation-01/sv-ddcc05p00-T0p05.wtxt"
    wdata = WData.load(inFile)

    # pprint.pp(vars(data))
    # pprint.pp(data.aliases)
    # pprint.pp(data.constants)

    # helpers:
    iteration = -1

    # plot:
    plot_current_density(
        wdata = wdata,
        iteration = iteration
    )

    return 0

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    sys.exit(main())
