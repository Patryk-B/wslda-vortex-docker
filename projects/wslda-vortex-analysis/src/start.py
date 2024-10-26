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
import numpy.typing as np_typing
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
        # grid:

        self.Nx: int = wdata.Nxyz[Axis.X]
        self.Ny: int = wdata.Nxyz[Axis.Y]

        self.dx: float = wdata.dxyz[Axis.X]
        self.dy: float = wdata.dxyz[Axis.Y]

        self.x: np.ndarray = wdata.xyz[Axis.X]
        self.y: np.ndarray = wdata.xyz[Axis.Y]

        self.x_flat: List[float] = self.x.flatten()
        self.y_flat: List[float] = self.y.flatten()

        # current density
        # - WARNING:
        #   - for matplotlib to display data corectly, wdata.j_<a|b>[<iteration>][<component>] needs to be transposed !!!
        #
        #     according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #     - wdata.j_a[iteration][Component.X][ Nx/2, :    ] === j_a_x( x = Nx/2, y        )
        #     - wdata.j_a[iteration][Component.X][ :   , Ny/2 ] === j_a_x( x       , y = Ny/2 )
        #
        #     therefore, after transposition:
        #     - parsed.j_a_x[ :   , Nx/2 ] === (wdata.j_a[iteration][Component.X].T)[ :   , Nx/2 ] === wdata.j_a[iteration][Component.X][ Nx/2, :    ] === j_a_x( x = Nx/2, y        )
        #     - parsed.j_a_x[ Ny/2, :    ] === (wdata.j_a[iteration][Component.X].T)[ Ny/2, :    ] === wdata.j_a[iteration][Component.X][ :   , Ny/2 ] === j_a_x( x       , y = Ny/2 )

        self.j_a_x: np.memmap = wdata.j_a[iteration][Component.X].T
        self.j_a_y: np.memmap = wdata.j_a[iteration][Component.Y].T
        self.j_a: np.ndarray = np.column_stack((self.j_a_x, self.j_a_y))
        self.j_a_mag: np.ndarray = np.sqrt(self.j_a_x ** 2 + self.j_a_y ** 2)

        self.j_b_x: np.memmap = wdata.j_b[iteration][Component.X].T
        self.j_b_y: np.memmap = wdata.j_b[iteration][Component.Y].T
        self.j_b: np.ndarray = np.column_stack((self.j_b_x, self.j_b_y))
        self.j_b_mag: np.ndarray = np.sqrt(self.j_b_x ** 2 + self.j_b_y ** 2)


class Plot():
    def __init__(
        self,
        rows_h_ratios: List[int],
        cols_w_ratios: List[int],
        subplot_h: int,
        subplot_w: int,
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


def gen_csecs_indices(
    half_point: float,
    csecs_as_precentages_before_half_point: List[float] = [],
    csecs_as_precentages_after_half_point: List[float] = [],
) ->List[int]:
    result: List[float] = []
    result += [np.ceil(half_point * precentage) for precentage in csecs_as_precentages_before_half_point]
    result += [half_point]
    result += [np.floor(half_point * precentage) for precentage in csecs_as_precentages_after_half_point]

    return [int(r) for r in result]


def gen_scientific_formatter() -> ticker.Formatter:
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2)) # Use scientific notation if <1e-2 or >1e2

    return formatter


# partial plots


def plot_streamlines(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,
    data_x: np_typing.ArrayLike,
    data_y: np_typing.ArrayLike,
    title: str,
    label_x: str,
    label_y: str,
) -> None:
    # Plot data:
    ax.streamplot(x, y, data_x, data_y, color='blue', linewidth=2, density=1.5)

    # Labels
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    # Axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    # # Aspect ratio:
    # ax.set_aspect('equal', adjustable='box')


def plot_pseudocolor(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,
    data: np_typing.ArrayLike,
    title: str,
    label_x: str,
    label_y: str,
) -> None:
    # Plot data:
    pcolor = ax.pcolormesh(x, y, data, cmap='plasma', shading='auto')

    # Labels
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    # Axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    # # Aspect ratio:
    # ax.set_aspect('equal', adjustable='box')

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


def plot_cross_sections_of_2d_data(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    cross_section_indices: List[int],
    x: np_typing.ArrayLike,
    data: np_typing.ArrayLike,
    gen_data_csec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],
    title: str,
    label_x: str,
    label_y: str,
    gen_label_data_func: Callable[[np_typing.ArrayLike, int], str],
) -> None:
    # The default color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot data:
    for i, color in zip(cross_section_indices, colors):

        # plot points:
        data_csec = gen_data_csec_func(data, i)
        ax.plot(
            x,
            data_csec,
            label=gen_label_data_func(title, x, i),
            marker='o',
            markersize=4,
            linestyle='None',
            color=color
        )

        # plot interpolation:
        int_x = np.linspace(x[0], x[-1], 200)
        int_data = interp1d(x, data_csec, kind='quadratic')
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

    # Scientific notation
    formatter = gen_scientific_formatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add legend
    ax.legend()


# plot systen properties:


def plot_current_density(
    data: ParsedWData,
    out_dir: str,
) -> None:
    p = Plot(
        cols_w_ratios = [1, 1, 1],
        rows_h_ratios = [1, 1, 1, 1, 1, 1, 1, 1],
        subplot_w = 10,
        subplot_h = 10,
    )

    x_csecs_00 = gen_csecs_indices(data.Nx/2)
    x_csecs_01 = gen_csecs_indices(data.Nx/2, [0.75, 0.875, 0.95])
    x_csecs_02 = gen_csecs_indices(data.Nx/2, [0.75, 0.875, 0.95], [1.05, 1.125, 1.25])

    y_csecs_00 = gen_csecs_indices(data.Ny/2)
    y_csecs_01 = gen_csecs_indices(data.Ny/2, [0.75, 0.875, 0.95])
    y_csecs_02 = gen_csecs_indices(data.Ny/2, [0.75, 0.875, 0.95], [1.05, 1.125, 1.25])

    get_data_x_csec = lambda data, x: data[:, x]
    get_data_y_csec = lambda data, y: data[y, :]

    get_data_x_csec_label = lambda title, x, i: f"{title}(x = {x[i]}, y)"
    get_data_y_csec_label = lambda title, y, i: f"{title}(x, y = {y[i]})"

    # plot data:

    plot_streamlines(p.fig, p.ax[0][1], data.x_flat, data.y_flat, data.j_a_x, data.j_a_y, 'j_a', 'x', 'y')

    plot_pseudocolor(p.fig, p.ax[1][0], data.x_flat, data.y_flat, data.j_a_mag, '||j_a||', 'x', 'y')
    plot_cross_sections_of_2d_data(p.fig, p.ax[1][1], y_csecs_00, data.x_flat, data.j_a_mag, get_data_y_csec, '||j_a||', 'x', '||j_a||', get_data_y_csec_label)
    plot_cross_sections_of_2d_data(p.fig, p.ax[1][2], x_csecs_00, data.y_flat, data.j_a_mag, get_data_x_csec, '||j_a||', 'y', '||j_a||', get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[2][0], data.x_flat, data.y_flat, data.j_a_x, 'j_a_x', 'x', 'y')
    plot_cross_sections_of_2d_data(p.fig, p.ax[2][1], y_csecs_02, data.x_flat, data.j_a_x, get_data_y_csec, 'j_a_x', 'x', 'j_a_x', get_data_y_csec_label)
    plot_cross_sections_of_2d_data(p.fig, p.ax[2][2], x_csecs_01, data.y_flat, data.j_a_x, get_data_x_csec, 'j_a_x', 'y', 'j_a_x', get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[3][0], data.x_flat, data.y_flat, data.j_a_y, 'j_a_y', 'x', 'y')
    plot_cross_sections_of_2d_data(p.fig, p.ax[3][1], y_csecs_01, data.x_flat, data.j_a_y, get_data_y_csec, 'j_a_y', 'x', 'j_a_y', get_data_y_csec_label)
    plot_cross_sections_of_2d_data(p.fig, p.ax[3][2], x_csecs_02, data.y_flat, data.j_a_y, get_data_x_csec, 'j_a_y', 'y', 'j_a_y', get_data_x_csec_label)

    plot_streamlines(p.fig, p.ax[4][1], data.x_flat, data.y_flat, data.j_b_x, data.j_b_y, 'j_b', 'x', 'y')

    plot_pseudocolor(p.fig, p.ax[5][0], data.x_flat, data.y_flat, data.j_b_mag, '||j_b||', 'x', 'y')
    plot_cross_sections_of_2d_data(p.fig, p.ax[5][1], y_csecs_00, data.x_flat, data.j_b_mag, get_data_y_csec, '||j_b||', 'x', '||j_b||', get_data_y_csec_label)
    plot_cross_sections_of_2d_data(p.fig, p.ax[5][2], x_csecs_00, data.y_flat, data.j_b_mag, get_data_x_csec, '||j_b||', 'y', '||j_b||', get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[6][0], data.x_flat, data.y_flat, data.j_b_x, 'j_b_x', 'x', 'y')
    plot_cross_sections_of_2d_data(p.fig, p.ax[6][1], y_csecs_02, data.x_flat, data.j_b_x, get_data_y_csec, 'j_b_x', 'x', 'j_b_x', get_data_y_csec_label)
    plot_cross_sections_of_2d_data(p.fig, p.ax[6][2], x_csecs_01, data.y_flat, data.j_b_x, get_data_x_csec, 'j_b_x', 'y', 'j_b_x', get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[7][0], data.x_flat, data.y_flat, data.j_b_y, 'j_b_y', 'x', 'y')
    plot_cross_sections_of_2d_data(p.fig, p.ax[7][1], y_csecs_01, data.x_flat, data.j_b_y, get_data_y_csec, 'j_b_y', 'x', 'j_b_y', get_data_y_csec_label)
    plot_cross_sections_of_2d_data(p.fig, p.ax[7][2], x_csecs_02, data.y_flat, data.j_b_y, get_data_x_csec, 'j_b_y', 'y', 'j_b_y', get_data_x_csec_label)

    # save plot:

    p.fig.tight_layout()
    p.fig.savefig(
        fname = out_dir + "/plot_current_density.png",
        dpi = 300
    )


# start:


def main() -> int:
    """
    entry point
    """

    in_dir = "/workspace/results/data"
    in_file = in_dir + "/st-vortex-recreation-01/sv-ddcc05p00-T0p05.wtxt"
    wdata = WData.load(in_file)
    parsed = ParsedWData(wdata, iteration=-1)

    out_dir = "/workspace/results/analysis"

    # plot:
    plot_current_density(
        data = parsed,
        out_dir = out_dir
    )

    return 0


if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    sys.exit(main())
