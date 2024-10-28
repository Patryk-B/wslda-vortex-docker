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

        self.nx: int = wdata.Nxyz[Axis.X]
        self.ny: int = wdata.Nxyz[Axis.Y]

        self.nx_half: int = int(self.nx/2)
        self.ny_half: int = int(self.ny/2)

        self.dx: float = wdata.dxyz[Axis.X]
        self.dy: float = wdata.dxyz[Axis.Y]

        self.x: np.ndarray = wdata.xyz[Axis.X]
        self.y: np.ndarray = wdata.xyz[Axis.Y]

        self.x_flat: List[float] = self.x.flatten()
        self.y_flat: List[float] = self.y.flatten()

        # rho (x,y)
        # - normal density (x,y)
        #
        # - NOTE:
        #   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #     - wdata.rho_a[iteration][ nx/2, :    ] === rho_a( x = nx/2, y        )
        #     - wdata.rho_a[iteration][ :   , ny/2 ] === rho_a( x       , y = ny/2 )

        self.rho_a: np.memmap = wdata.rho_a[iteration]
        self.rho_b: np.memmap = wdata.rho_b[iteration]
        self.rho_tot: np.memmap = self.rho_a + self.rho_b

        # delta (x,y)
        # - pairing gap function (x,y)
        #
        # - NOTE:
        #   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #     - wdata.delta[iteration][ nx/2, :    ] === delta( x = nx/2, y        )
        #     - wdata.delta[iteration][ :   , ny/2 ] === delta( x       , y = ny/2 )

        self.delta = wdata.delta[iteration]
        self.delta_norm = np.abs(self.delta)

        # j (x,y)
        # - current density (x,y)
        #
        # - WARNING:
        #   - in sone places j needs to be transposed !!!
        #
        # - NOTE:
        #   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #     - wdata.j_a[iteration][Component.X][ nx/2, :    ] === j_a_x( x = nx/2, y        )
        #     - wdata.j_a[iteration][Component.X][ :   , ny/2 ] === j_a_x( x       , y = ny/2 )

        self.j_a_x: np.memmap = wdata.j_a[iteration][Component.X]
        self.j_a_y: np.memmap = wdata.j_a[iteration][Component.Y]
        self.j_a: np.ndarray = np.column_stack((self.j_a_x, self.j_a_y))
        self.j_a_norm: np.ndarray = np.sqrt(self.j_a_x ** 2 + self.j_a_y ** 2)

        self.j_b_x: np.memmap = wdata.j_b[iteration][Component.X]
        self.j_b_y: np.memmap = wdata.j_b[iteration][Component.Y]
        self.j_b: np.ndarray = np.column_stack((self.j_b_x, self.j_b_y))
        self.j_b_norm: np.ndarray = np.sqrt(self.j_b_x ** 2 + self.j_b_y ** 2)

        self.j_tot_x: np.memmap = self.j_a_x + self.j_b_x
        self.j_tot_y: np.memmap = self.j_a_y + self.j_b_y
        self.j_tot: np.ndarray = np.column_stack((self.j_tot_x, self.j_tot_y))
        self.j_tot_norm: np.ndarray = np.sqrt(self.j_tot_x ** 2 + self.j_tot_y ** 2)

        #
        # NOTE:
        # - vortex core lies in the center of the grid at cords [ x[nx/2], y[ny/2] ] == [40, 40]
        # - therefore assuming vortex bulk spans a circle with r == x[nx/4] == y[ny/4] == 20
        #

        bulk_radius = self.nx_half / 2
        epsilon = 0.01

        # rho_v (x = 0, y = 0)
        # - vortex normal density
        # - normal density at the center of the vortex core

        self.rho_tot_v: float = self.rho_tot[self.nx_half, self.ny_half]
        self.rho_a_v: float = self.rho_a[self.nx_half, self.ny_half]
        self.rho_b_v: float = self.rho_b[self.nx_half, self.ny_half]

        # rho_0 (x > rv, y > rv)
        # - bulk normal density
        # - normal density far from the center of the vortex core

        self.rho_tot_0: float = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.rho_tot,
            bulk_radius,
            epsilon
        )
        self.rho_a_0: float = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.rho_a,
            bulk_radius,
            epsilon
        )
        self.rho_b_0: float = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.rho_b,
            bulk_radius,
            epsilon
        )

        # delta_0 (x > rv, y > rv)
        # - bulk pairing gap function
        # - pairing gap function far from the center of the vortex core

        self.delta_norm_0: float = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.delta_norm,
            bulk_radius,
            epsilon
        )

        # j_0 (x > rv, y > rv)
        # - bulk current density
        # - current density far from the center of the vortex core

        self.j_a_norm_0: float = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.j_a_norm,
            bulk_radius,
            epsilon
        )
        self.j_b_norm_0: float = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.j_b_norm,
            bulk_radius,
            epsilon
        )
        self.j_tot_norm_0: float = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.j_tot_norm,
            bulk_radius,
            epsilon
        )


    def __calc_arithmetic_mean_over_circle(
        self,
        x: List[float],
        y: List[float],
        data: np.ndarray,
        R: float,
        epsilon: float
    ):
        # Check which indices lie on a circle with radius R, with precision epsilon:
        indices = []
        for i in range(0, len(x)):
            for j in range(0, len(y)):
                r = np.sqrt(x[i] ** 2 + y[j] ** 2)

                if np.abs(r - R) <= epsilon:
                    indices.append([i, j])

        # calc data's arithmetic mean value on the circle:
        result = 0
        for i, j in indices:
            # print (f"{x[i]}, {y[j]} ---> r = {np.sqrt(x[i] ** 2 + y[j] ** 2)}")
            result += data[i, j]
        result /= len(indices)

        return result


class Plot():
    def __init__(
        self,
        subplot_h: int,
        subplot_w: int,
        rows_h_ratios: List[int],
        cols_w_ratios: List[int],
        ignore_subplots: List[List[int]] = []
    ):
        # plot's width and height:

        self.subplot_h: float = subplot_h
        self.subplot_w: float = subplot_w

        self.nrows: float = len(rows_h_ratios)
        self.ncols: float = len(cols_w_ratios)

        self.rows_h_ratios: List[float] = rows_h_ratios
        self.cols_w_ratios: List[float] = cols_w_ratios

        self.h: float = np.sum(self.subplot_h * np.array(self.rows_h_ratios))
        self.w: float = np.sum(self.subplot_w * np.array(self.cols_w_ratios))

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

        self.ignored_subplots: List[List[int]] = ignore_subplots

        self.ax: List[List[matplotlib.axes.Axes]] = [[None for _ in range(self.ncols)] for _ in range(self.nrows)]
        for i in range(0, self.nrows):
            for j in range(0, self.ncols):
                if [i, j] not in self.ignored_subplots:
                    self.ax[i][j] = self.fig.add_subplot(self.gs[i, j])


def gen_csecs_indices(
    half_point: float,
    csecs_as_precentages_before_half_point: List[float] = [],
    csecs_as_precentages_after_half_point: List[float] = [],
) ->List[int]:
    return [
        int(i) for i in [
            np.ceil(half_point * precentage) for precentage in csecs_as_precentages_before_half_point
        ] + [
            half_point
        ] + [
            np.floor(half_point * precentage) for precentage in csecs_as_precentages_after_half_point
        ]
    ]


def gen_default_color_cycle() -> List[str]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def gen_scientific_formatter() -> ticker.Formatter:
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2)) # Use scientific notation if <1e-2 or >1e2

    return formatter


# basic plots


def plot_vectors(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,
    data_x: np_typing.ArrayLike,
    data_y: np_typing.ArrayLike,
    stride: int,
    scale: float,
    title: str,
    label_x: str,
    label_y: str,
) -> None:
    # Create meshgrid
    # - redefines x and y as 2d arrays becasue; x and y argunets are 1d arrays while data_x and data_y are 2d arrays
    grid_x, grid_y = np.meshgrid(x, y)

    # Apply slicing to reduce vector density
    sliced_x = grid_x[::stride, ::stride]
    sliced_y = grid_y[::stride, ::stride]
    sliced_u = data_x[::stride, ::stride]
    sliced_v = data_y[::stride, ::stride]

    # The default color cycle
    colors = gen_default_color_cycle()[0:3]

    # Plot data:
    ax.quiver(sliced_x, sliced_y, sliced_u, sliced_v, angles='xy', scale_units='xy', scale=scale, color=colors)

    # Labels
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    # Axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    # Aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Grid lines
    ax.grid()
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)


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
    # The default color cycle
    color = gen_default_color_cycle()[0]

    # Plot data:
    ax.streamplot(x, y, data_x, data_y, color=color, linewidth=1, density=1.5)

    # Labels
    ax.set_title(title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    # Axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    # Grid lines
    ax.grid()
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)

    # Aspect ratio:
    ax.set_aspect('equal', adjustable='box')


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

    # # Grid lines
    # ax.grid()
    # ax.axhline(0, color='grey', lw=0.5)
    # ax.axvline(0, color='grey', lw=0.5)

    # Aspect ratio:
    ax.set_aspect('equal', adjustable='box')

    # Add color bar
    color_bar = fig.colorbar(
        pcolor,
        ax = ax,
        location = 'right',
        orientation = 'vertical',
        fraction = 0.035,
        pad = 0.01
    )

    # Set scientific notation for color bar labels
    formatter = gen_scientific_formatter()
    color_bar.ax.yaxis.set_major_formatter(formatter)

    # Move the offset position (scientific notation) to the right
    color_bar.ax.yaxis.offsetText.set_x(2)


# basic plots tailored to data


def plot_csecs_of_2d_data(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    csec_indices: List[int],
    x: np_typing.ArrayLike,
    data: np_typing.ArrayLike,
    gen_data_csec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],
    title: str,
    label_x: str,
    label_y: str,
    labe_legend: str,
    gen_legend_label_func: Callable[[np_typing.ArrayLike, int], str],
) -> None:
    # The default color cycle
    colors = gen_default_color_cycle()

    # Plot data:
    for i, color in zip(csec_indices, colors):

        # plot points:
        data_csec = gen_data_csec_func(data, i)
        ax.plot(
            x,
            data_csec,
            label=gen_legend_label_func(labe_legend, x, i),
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

    # Grid lines
    ax.grid()
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)

    # # Aspect ratio:
    # ax.set_aspect('equal', adjustable='box')

    # Scientific notation
    formatter = gen_scientific_formatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add legend
    ax.legend()


# plot systen properties:


def plot_normal_density(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: List[float],
    y: List[float],

    rho_tot: np.ndarray,
    rho_a: np.ndarray,
    rho_b: np.ndarray,

    title_rho_tot: str,
    title_rho_a: str,
    title_rho_b: str,

    lable_x: str,
    lable_y: str,

    lable_rho_tot: str,
    lable_rho_a: str,
    lable_rho_b: str,

    legend_rho_tot: str,
    legend_rho_a: str,
    legend_rho_b: str,

) -> None:
    p = Plot(
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        cols_w_ratios = [1.11, 1, 1],
        rows_h_ratios = [1, 1, 1],
    )

    x_csecs_00 = gen_csecs_indices(nx/2)
    y_csecs_00 = gen_csecs_indices(ny/2)

    get_data_x_csec = lambda data, x: data[x, :]
    get_data_y_csec = lambda data, y: data[:, y]

    get_data_x_csec_label = lambda legend, x, i: f"{legend}(x = {x[i]}, y)"
    get_data_y_csec_label = lambda legend, y, i: f"{legend}(x, y = {y[i]})"

    # plot data:
    # - rho

    plot_pseudocolor(p.fig, p.ax[0][0], x, y, rho_tot, title_rho_tot, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[0][1], y_csecs_00, x, rho_tot, get_data_y_csec, title_rho_tot, lable_x, lable_rho_tot, legend_rho_tot, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[0][2], x_csecs_00, y, rho_tot, get_data_x_csec, title_rho_tot, lable_y, lable_rho_tot, legend_rho_tot, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[1][0], x, y, rho_a, title_rho_a, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[1][1], y_csecs_00, x, rho_a, get_data_y_csec, title_rho_a, lable_x, lable_rho_a, legend_rho_a, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[1][2], x_csecs_00, y, rho_a, get_data_x_csec, title_rho_a, lable_y, lable_rho_a, legend_rho_a, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[2][0], x, y, rho_b, title_rho_b, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[2][1], y_csecs_00, x, rho_b, get_data_y_csec, title_rho_b, lable_x, lable_rho_b, legend_rho_b, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[2][2], x_csecs_00, y, rho_b, get_data_x_csec, title_rho_b, lable_y, lable_rho_b, legend_rho_b, get_data_x_csec_label)

    # save plot:

    p.fig.tight_layout()
    p.fig.savefig(fname = out_file, dpi = dpi)


def plot_pairing_gap_function(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: List[float],
    y: List[float],

    delta_norm: np.ndarray,

    title_delta_norm: str,

    lable_x: str,
    lable_y: str,

    lable_delta_norm: str,

    legend_delta_norm: str,

) -> None:
    p = Plot(
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        cols_w_ratios = [1.11, 1, 1],
        rows_h_ratios = [1],
    )

    x_csecs_00 = gen_csecs_indices(nx/2)
    y_csecs_00 = gen_csecs_indices(ny/2)

    get_data_x_csec = lambda data, x: data[x, :]
    get_data_y_csec = lambda data, y: data[:, y]

    get_data_x_csec_label = lambda legend, x, i: f"{legend}(x = {x[i]}, y)"
    get_data_y_csec_label = lambda legend, y, i: f"{legend}(x, y = {y[i]})"

    # plot data:
    # - delta

    plot_pseudocolor(p.fig, p.ax[0][0], x, y, delta_norm, title_delta_norm, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[0][1], y_csecs_00, x, delta_norm, get_data_y_csec, title_delta_norm, lable_x, lable_delta_norm, legend_delta_norm, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[0][2], x_csecs_00, y, delta_norm, get_data_x_csec, title_delta_norm, lable_y, lable_delta_norm, legend_delta_norm, get_data_x_csec_label)

    # save plot:

    p.fig.tight_layout()
    p.fig.savefig(fname = out_file, dpi = dpi)


def plot_current_density(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: List[float],
    y: List[float],

    j_tot_norm: np.ndarray,
    j_tot_x: np.ndarray,
    j_tot_y: np.ndarray,

    j_a_norm: np.ndarray,
    j_a_x: np.ndarray,
    j_a_y: np.ndarray,

    j_b_norm: np.ndarray,
    j_b_x: np.ndarray,
    j_b_y: np.ndarray,

    title_j_tot: str,
    title_j_tot_norm: str,
    title_j_tot_x: str,
    title_j_tot_y: str,

    title_j_a: str,
    title_j_a_norm: str,
    title_j_a_x: str,
    title_j_a_y: str,

    title_j_b: str,
    title_j_b_norm: str,
    title_j_b_x: str,
    title_j_b_y: str,

    lable_x: str,
    lable_y: str,

    label_j_tot_norm: str,
    label_j_tot_x: str,
    label_j_tot_y: str,

    label_j_a_norm: str,
    label_j_a_x: str,
    label_j_a_y: str,

    label_j_b_norm: str,
    label_j_b_x: str,
    label_j_b_y: str,

    legend_j_tot_norm: str,
    legend_j_tot_x: str,
    legend_j_tot_y: str,

    legend_j_a_norm: str,
    legend_j_a_x: str,
    legend_j_a_y: str,

    legend_j_b_norm: str,
    legend_j_b_x: str,
    legend_j_b_y: str,

) -> None:
    p = Plot(
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        cols_w_ratios = [1.11, 1, 1],
        rows_h_ratios = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ignore_subplots = [
            [0, 2],
            [4, 2],
            [8, 2]
        ]
    )

    x_csecs_00 = gen_csecs_indices(nx/2)
    x_csecs_01 = gen_csecs_indices(nx/2, [0.75, 0.875, 0.95])
    x_csecs_02 = gen_csecs_indices(nx/2, [0.75, 0.875, 0.95], [1.05, 1.125, 1.25])

    y_csecs_00 = gen_csecs_indices(ny/2)
    y_csecs_01 = gen_csecs_indices(ny/2, [0.75, 0.875, 0.95])
    y_csecs_02 = gen_csecs_indices(ny/2, [0.75, 0.875, 0.95], [1.05, 1.125, 1.25])

    get_data_x_csec = lambda data, x: data[x, :]
    get_data_y_csec = lambda data, y: data[:, y]

    get_data_x_csec_label = lambda legend, x, i: f"{legend}(x = {x[i]}, y)"
    get_data_y_csec_label = lambda legend, y, i: f"{legend}(x, y = {y[i]})"

    vector_stride = 3
    vector_scale = 0.0005

    # WARNING:
    # - for matplotlib to plot 2d plots of density current correctly, in sone places wj_<a|b>[<iteration>][<component>] needs to be transposed !!!

    TRANSPOSE: Callable[[np_typing.ArrayLike], np_typing.ArrayLike] = lambda data: data.T

    # plot data:
    # - j_total

    plot_vectors(p.fig, p.ax[0][0], x, y, TRANSPOSE(j_tot_x), TRANSPOSE(j_tot_y), vector_stride, vector_scale, title_j_tot, lable_x, lable_y)
    plot_streamlines(p.fig, p.ax[0][1], x, y, TRANSPOSE(j_tot_x), TRANSPOSE(j_tot_y), title_j_tot, lable_x, lable_y)

    plot_pseudocolor(p.fig, p.ax[1][0], x, y, TRANSPOSE(j_tot_norm), title_j_tot_norm, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[1][1], y_csecs_00, x, j_tot_norm, get_data_y_csec, title_j_tot_norm, lable_x, label_j_tot_norm, legend_j_tot_norm, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[1][2], x_csecs_00, y, j_tot_norm, get_data_x_csec, title_j_tot_norm, lable_y, label_j_tot_norm, legend_j_tot_norm, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[2][0], x, y, TRANSPOSE(j_tot_x), title_j_tot_x, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[2][1], y_csecs_02, x, j_tot_x, get_data_y_csec, title_j_tot_x, lable_x, label_j_tot_x, legend_j_tot_x, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[2][2], x_csecs_01, y, j_tot_x, get_data_x_csec, title_j_tot_x, lable_y, label_j_tot_x, legend_j_tot_x, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[3][0], x, y, TRANSPOSE(j_tot_y), title_j_tot_y, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[3][1], y_csecs_01, x, j_tot_y, get_data_y_csec, title_j_tot_y, lable_x, label_j_tot_y, legend_j_tot_y, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[3][2], x_csecs_02, y, j_tot_y, get_data_x_csec, title_j_tot_y, lable_y, label_j_tot_y, legend_j_tot_y, get_data_x_csec_label)

    # plot data:
    # - j_a

    plot_vectors(p.fig, p.ax[4][0], x, y, TRANSPOSE(j_a_x), TRANSPOSE(j_a_y), vector_stride, vector_scale, title_j_a, lable_x, lable_y)
    plot_streamlines(p.fig, p.ax[4][1], x, y, TRANSPOSE(j_a_x), TRANSPOSE(j_a_y), title_j_a, lable_x, lable_y)

    plot_pseudocolor(p.fig, p.ax[5][0], x, y, TRANSPOSE(j_a_norm), title_j_a_norm, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[5][1], y_csecs_00, x, j_a_norm, get_data_y_csec, title_j_a_norm, lable_x, label_j_a_norm, legend_j_a_norm, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[5][2], x_csecs_00, y, j_a_norm, get_data_x_csec, title_j_a_norm, lable_y, label_j_a_norm, legend_j_a_norm, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[6][0], x, y, TRANSPOSE(j_a_x), title_j_a_x, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[6][1], y_csecs_02, x, j_a_x, get_data_y_csec, title_j_a_x, lable_x, label_j_a_x, legend_j_a_x, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[6][2], x_csecs_01, y, j_a_x, get_data_x_csec, title_j_a_x, lable_y, label_j_a_x, legend_j_a_x, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[7][0], x, y, TRANSPOSE(j_a_y), title_j_a_y, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[7][1], y_csecs_01, x, j_a_y, get_data_y_csec, title_j_a_y, lable_x, label_j_a_y, legend_j_a_y, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[7][2], x_csecs_02, y, j_a_y, get_data_x_csec, title_j_a_y, lable_y, label_j_a_y, legend_j_a_y, get_data_x_csec_label)

    # plot data:
    # - j_b

    plot_vectors(p.fig, p.ax[8][0], x, y, TRANSPOSE(j_b_x), TRANSPOSE(j_b_y), vector_stride, vector_scale, title_j_b, lable_x, lable_y)
    plot_streamlines(p.fig, p.ax[8][1], x, y, TRANSPOSE(j_b_x), TRANSPOSE(j_b_y), title_j_b, lable_x, lable_y)

    plot_pseudocolor(p.fig, p.ax[9][0], x, y, TRANSPOSE(j_b_norm), title_j_b_norm, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[9][1], y_csecs_00, x, j_b_norm, get_data_y_csec, title_j_b_norm, lable_x, label_j_b_norm, legend_j_b_norm, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[9][2], x_csecs_00, y, j_b_norm, get_data_x_csec, title_j_b_norm, lable_y, label_j_b_norm, legend_j_b_norm, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[10][0], x, y, TRANSPOSE(j_b_x), title_j_b_x, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[10][1], y_csecs_02, x, j_b_x, get_data_y_csec, title_j_b_x, lable_x, label_j_b_x, legend_j_b_x, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[10][2], x_csecs_01, y, j_b_x, get_data_x_csec, title_j_b_x, lable_y, label_j_b_x, legend_j_b_x, get_data_x_csec_label)

    plot_pseudocolor(p.fig, p.ax[11][0], x, y, TRANSPOSE(j_b_y), title_j_b_y, lable_x, lable_y)
    plot_csecs_of_2d_data(p.fig, p.ax[11][1], y_csecs_01, x, j_b_y, get_data_y_csec, title_j_b_y, lable_x, label_j_b_y, legend_j_b_y, get_data_y_csec_label)
    plot_csecs_of_2d_data(p.fig, p.ax[11][2], x_csecs_02, y, j_b_y, get_data_x_csec, title_j_b_y, lable_y, label_j_b_y, legend_j_b_y, get_data_x_csec_label)

    # save plot:

    p.fig.tight_layout()
    p.fig.savefig(fname = out_file, dpi = dpi)


# start:


def main() -> int:
    """
    entry point
    """

    in_dir = "/workspace/results/data"
    in_file = in_dir + "/st-vortex-recreation-01/sv-ddcc05p00-T0p05.wtxt"
    wdata = WData.load(in_file)

    iteration = -1
    parsed = ParsedWData(wdata, iteration)

    subplot_w = 7
    subplot_h = 7

    out_dir = "/workspace/results/analysis"
    dpi = 300

    # plot:

    plot_normal_density(
        out_file = out_dir + "/plot_normal_densit.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        rho_tot = parsed.rho_tot,
        rho_a = parsed.rho_a,
        rho_b = parsed.rho_b,

        title_rho_tot = 'rho_tot',
        title_rho_a = 'rho_a',
        title_rho_b = 'rho_b',

        lable_x = 'x',
        lable_y = 'y',

        lable_rho_tot = 'rho_tot',
        lable_rho_a = 'rho_a',
        lable_rho_b = 'rho_b',

        legend_rho_tot = 'rho_tot',
        legend_rho_a = 'rho_a',
        legend_rho_b = 'rho_b',
    )

    plot_normal_density(
        out_file = out_dir + "/plot_normal_densit_normalized.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        rho_tot = parsed.rho_tot / parsed.rho_tot_0,
        rho_a = parsed.rho_a / parsed.rho_a_0,
        rho_b = parsed.rho_b / parsed.rho_b_0,

        title_rho_tot = 'rho_total / rho_total_0',
        title_rho_a = 'rho_a / rho_a_0',
        title_rho_b = 'rho_b / rho_b_0',

        lable_x = 'x',
        lable_y = 'y',

        lable_rho_tot = 'rho_total / rho_total_0',
        lable_rho_a = 'rho_a / rho_a_0',
        lable_rho_b = 'rho_b / rho_b_0',

        legend_rho_tot = 'rho_total / rho_total_0 ',
        legend_rho_a = 'rho_a / rho_a_0 ',
        legend_rho_b = 'rho_b / rho_b_0 ',
    )

    plot_pairing_gap_function(
        out_file = out_dir + "/plot_pairing_gap_function.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        delta_norm = parsed.delta_norm,

        title_delta_norm = '|Δ|',

        lable_x = 'x',
        lable_y = 'y',

        lable_delta_norm = '|Δ|',

        legend_delta_norm = '|Δ|',
    )

    plot_pairing_gap_function(
        out_file = out_dir + "/plot_pairing_gap_function_normalized.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        delta_norm = parsed.delta_norm / parsed.delta_norm_0,

        title_delta_norm = '|Δ| / |Δ_0|',

        lable_x = 'x',
        lable_y = 'y',

        lable_delta_norm = '|Δ| / |Δ_0|',

        legend_delta_norm = '|Δ| / |Δ_0| ',
    )

    plot_current_density(
        out_file = out_dir + "/plot_current_density.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        j_tot_norm = parsed.j_tot_norm,
        j_tot_x = parsed.j_tot_x,
        j_tot_y = parsed.j_tot_y,

        j_a_norm = parsed.j_a_norm,
        j_a_x = parsed.j_a_x,
        j_a_y = parsed.j_a_y,

        j_b_norm = parsed.j_b_norm,
        j_b_x = parsed.j_b_x,
        j_b_y = parsed.j_b_y,

        title_j_tot = 'j_total',
        title_j_tot_norm = '|j_total|',
        title_j_tot_x = 'j_total_x',
        title_j_tot_y = 'j_total_y',

        title_j_a = 'j_a',
        title_j_a_norm = '|j_a|',
        title_j_a_x = 'j_a_x',
        title_j_a_y = 'j_a_y',

        title_j_b = 'j_b',
        title_j_b_norm = '|j_b|',
        title_j_b_x = 'j_b_x',
        title_j_b_y = 'j_b_y',

        lable_x = 'x',
        lable_y = 'y',

        label_j_tot_norm = '|j_total|',
        label_j_tot_x = 'j_total_x',
        label_j_tot_y = 'j_total_y',

        label_j_a_norm = '|j_a|',
        label_j_a_x = 'j_a_x',
        label_j_a_y = 'j_a_y',

        label_j_b_norm = '|j_b|',
        label_j_b_x = 'j_b_x',
        label_j_b_y = 'j_b_y',

        legend_j_tot_norm = '|j_total|',
        legend_j_tot_x = 'j_total_x',
        legend_j_tot_y = 'j_total_y',

        legend_j_a_norm = '|j_a|',
        legend_j_a_x = 'j_a_x',
        legend_j_a_y = 'j_a_y',

        legend_j_b_norm = '|j_b|',
        legend_j_b_x = 'j_b_x',
        legend_j_b_y = 'j_b_y',
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
