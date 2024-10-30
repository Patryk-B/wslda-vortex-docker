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

        print(f"nx = {self.nx}")
        print(f"ny = {self.ny}")
        print()

        self.nx_half: int = int(self.nx/2)
        self.ny_half: int = int(self.ny/2)

        print(f"nx_half = {self.nx_half}")
        print(f"ny_half = {self.ny_half}")
        print()

        self.nx_quarter: int = int(self.nx_half/2)
        self.ny_quarter: int = int(self.ny_half/2)

        print(f"nx_quarter = {self.nx_quarter}")
        print(f"ny_quarter = {self.ny_quarter}")
        print()

        self.dx: float = wdata.dxyz[Axis.X]
        self.dy: float = wdata.dxyz[Axis.Y]

        print(f"dx = {self.dx}")
        print(f"dy = {self.dy}")
        print()

        self.x: np.ndarray = wdata.xyz[Axis.X]
        self.y: np.ndarray = wdata.xyz[Axis.Y]

        self.x_flat: np.ndarray = self.x.flatten()
        self.y_flat: np.ndarray = self.y.flatten()

        # radius:
        # - center of the vortex core lies at the center of the grid
        #   r_v = x[nx * 1/2]
        #       = y[ny * 1/2]
        #       = 0
        #
        # - vortex bulk spans a circle with r = r_0
        #   r_0 = abs(x[nx * 1/4]) =
        #       = abs(x[nx * 3/4]) =
        #       = y[ny * 3/4]      =
        #       = 20

        self.x_index_v = self.nx_half
        self.x_index_0 = self.nx_half + self.nx_quarter

        self.y_index_v = self.ny_half
        self.y_index_0 = self.ny_half + self.ny_quarter

        print(f"x_index_v = {self.x_index_v}")
        print(f"x_index_0 = {self.x_index_0}")
        print()
        print(f"y_index_v = {self.y_index_v}")
        print(f"y_index_0 = {self.y_index_0}")
        print()

        self.x_v = self.x_flat[self.x_index_v]
        self.x_0 = self.x_flat[self.x_index_0]

        self.y_v = self.y_flat[self.y_index_v]
        self.y_0 = self.y_flat[self.y_index_0]

        print(f"x_v = {self.x_v}")
        print(f"x_0 = {self.x_0}")
        print()
        print(f"y_v = {self.y_v}")
        print(f"y_0 = {self.y_0}")
        print()

        self.r_v = self.x_v
        self.r_0 = self.x_0
        self.r_epsilon = 0.01,

        print(f"r_v = {self.r_v}")
        print(f"r_0 = {self.r_0}")
        print(f"r_epsilon = {self.r_epsilon}")
        print()

        # rho (x,y) -> Real
        # - normal density (x,y)
        #
        # - NOTE:
        #   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #     - wdata.rho_a[iteration][ nx/2, :    ] === rho_a( x = nx/2, y        )
        #     - wdata.rho_a[iteration][ :   , ny/2 ] === rho_a( x       , y = ny/2 )

        self.rho_a: np.memmap = wdata.rho_a[iteration]
        self.rho_b: np.memmap = wdata.rho_b[iteration]
        self.rho_tot: np.memmap = self.rho_a + self.rho_b

        # rho_v = <rho(r = r_v = 0)>
        # - vortex normal density
        # - normal density at the center of the vortex core

        self.rho_a_v: np.float64 = self.rho_a[self.nx_half, self.ny_half]
        self.rho_b_v: np.float64 = self.rho_b[self.nx_half, self.ny_half]
        self.rho_tot_v: np.float64 = self.rho_tot[self.nx_half, self.ny_half]

        # rho_0 = <rho(r = r_0)>
        # - bulk normal density
        # - normal density far from the center of the vortex core

        self.rho_a_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.rho_a,
            self.r_0,
            self.r_epsilon
        )
        self.rho_b_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.rho_b,
            self.r_0,
            self.r_epsilon
        )
        self.rho_tot_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.rho_tot,
            self.r_0,
            self.r_epsilon
        )

        # delta (x,y) -> Conplex
        # - pairing gap function (x,y)
        #
        # - NOTE:
        #   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #     - wdata.delta[iteration][ nx/2, :    ] === delta( x = nx/2, y        )
        #     - wdata.delta[iteration][ :   , ny/2 ] === delta( x       , y = ny/2 )

        self.delta = wdata.delta[iteration]
        self.delta_norm = np.abs(self.delta)

        # delta_norm_0 = <delta_norm(r = r_0)>
        # - bulk pairing gap function
        # - pairing gap function far from the center of the vortex core

        self.delta_norm_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.delta_norm,
            self.r_0,
            self.r_epsilon
        )

        # j (x,y) -> Vector
        # - current density (x,y)
        #
        # - WARNING:
        #   - j needs to be transposed !!!
        #
        # - NOTE:
        #   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #     - wdata.j_a[iteration][Component.X][ nx/2, :    ] === j_a_x( x = nx/2, y        )
        #     - wdata.j_a[iteration][Component.X][ :   , ny/2 ] === j_a_x( x       , y = ny/2 )

        self.j_a_x: np.memmap = wdata.j_a[iteration][Component.X].T
        self.j_a_y: np.memmap = wdata.j_a[iteration][Component.Y].T
        self.j_a: np.ndarray = np.column_stack((self.j_a_x, self.j_a_y))
        self.j_a_norm: np.ndarray = np.sqrt(self.j_a_x ** 2 + self.j_a_y ** 2)

        self.j_b_x: np.memmap = wdata.j_b[iteration][Component.X].T
        self.j_b_y: np.memmap = wdata.j_b[iteration][Component.Y].T
        self.j_b: np.ndarray = np.column_stack((self.j_b_x, self.j_b_y))
        self.j_b_norm: np.ndarray = np.sqrt(self.j_b_x ** 2 + self.j_b_y ** 2)

        self.j_tot_x: np.memmap = self.j_a_x + self.j_b_x
        self.j_tot_y: np.memmap = self.j_a_y + self.j_b_y
        self.j_tot: np.ndarray = np.column_stack((self.j_tot_x, self.j_tot_y))
        self.j_tot_norm: np.ndarray = np.sqrt(self.j_tot_x ** 2 + self.j_tot_y ** 2)

        # j_0 = <j(r = r_0)>
        # - bulk current density
        # - current density far from the center of the vortex core

        self.j_a_norm_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.j_a_norm,
            self.r_0,
            self.r_epsilon
        )
        self.j_b_norm_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.j_b_norm,
            self.r_0,
            self.r_epsilon
        )
        self.j_tot_norm_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.j_tot_norm,
            self.r_0,
            self.r_epsilon
        )

        # nu (x,y) -> Complex
        # -

        self.nu: np.memmap = wdata.nu[iteration]

        # tau (x,y) -> Real
        # -

        self.tau_a: np.memmap = wdata.tau_a[iteration]
        self.tau_b: np.memmap = wdata.tau_b[iteration]
        self.tau_tot: np.memmap = self.tau_a + self.tau_b

        # V (x,y) -> Real
        # -

        self.V_a: np.memmap = wdata.V_a[iteration]
        self.V_b: np.memmap = wdata.V_b[iteration]
        self.V_tot: np.memmap = self.V_a + self.V_b

        # V_ext (x,y) -> Real
        # -

        self.V_ext_a: np.memmap = wdata.V_ext_a[iteration]
        self.V_ext_b: np.memmap = wdata.V_ext_b[iteration]
        self.V_ext_tot: np.memmap = self.V_ext_a + self.V_ext_b

        # delta_ext (x,y) -> Complex
        # -

        self.delta_ext: np.memmap = wdata.delta_ext[iteration]

        # velocity_ext (x,y) -> Vector
        # -

        self.velocity_ext_a_x: np.memmap = wdata.velocity_ext_a[iteration][Component.X]
        self.velocity_ext_a_y: np.memmap = wdata.velocity_ext_a[iteration][Component.Y]
        self.velocity_ext_a: np.ndarray = np.column_stack((self.velocity_ext_a_x, self.velocity_ext_a_y))
        self.velocity_ext_a_norm: np.ndarray = np.sqrt(self.velocity_ext_a_x ** 2 + self.velocity_ext_a_y ** 2)

        self.velocity_ext_b_x: np.memmap = wdata.velocity_ext_b[iteration][Component.X]
        self.velocity_ext_b_y: np.memmap = wdata.velocity_ext_b[iteration][Component.Y]
        self.velocity_ext_b: np.ndarray = np.column_stack((self.velocity_ext_b_x, self.velocity_ext_b_y))
        self.velocity_ext_b_norm: np.ndarray = np.sqrt(self.velocity_ext_b_x ** 2 + self.velocity_ext_b_y ** 2)

        self.velocity_ext_tot_x: np.memmap = self.velocity_ext_a_x + self.velocity_ext_b_x
        self.velocity_ext_tot_y: np.memmap = self.velocity_ext_a_y + self.velocity_ext_b_y
        self.velocity_ext_tot: np.ndarray = np.column_stack((self.velocity_ext_tot_x, self.velocity_ext_tot_y))
        self.velocity_ext_tot_norm: np.ndarray = np.sqrt(self.velocity_ext_tot_x ** 2 + self.velocity_ext_tot_y ** 2)

        # alpha (x,y) -> Real
        # -

        self.alpha_a: np.memmap = wdata.alpha_a[iteration]
        self.alpha_b: np.memmap = wdata.alpha_b[iteration]
        self.alpha_tot: np.memmap = self.alpha_a + self.alpha_b

        # alpha_0 = <alpha(r_0)>
        # -

        self.alpha_a_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.alpha_a,
            self.r_0,
            self.r_epsilon
        )
        self.alpha_b_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.alpha_b,
            self.r_0,
            self.r_epsilon
        )
        self.alpha_tot_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.x_flat,
            self.y_flat,
            self.alpha_tot,
            self.r_0,
            self.r_epsilon
        )

        # A (x,y) -> Vector
        # -

        self.A_a: np.memmap = wdata.A_a[iteration]
        self.A_b: np.memmap = wdata.A_b[iteration]
        self.A_tot: np.memmap = self.A_a + self.A_b

        # k_F
        # - fermi momentum
        # - calcualted from data_0

        self.k_F: np.float64 = np.cbrt(3.0 * (np.pi ** 2) * self.rho_tot_0)
        pprint.pp(self.k_F)

        # epislon_F
        # - fermi energy
        # - calcualted from data_0

        self.epsilon_F: np.float64 = (self.k_F ** 2) / 2.0
        pprint.pp(self.epsilon_F)

        # ass_star
        # - effective ass

        # self.A_0 =

        # pprint.pp(wdata.alpha_a)

        # epison_star

        # l_c
        # - coherence length

    def __calc_arithmetic_mean_over_circle(
        self,
        x: List[float],
        y: List[float],
        data: np.ndarray,
        R: float,
        epsilon: float = 0.01
    ) -> np.float64:
        # Check which indices lie on a circle with radius R, with precision epsilon:
        indices = []
        for i in range(0, len(x)):
            for j in range(0, len(y)):
                r = np.sqrt(x[i] ** 2 + y[j] ** 2)

                if np.abs(r - R) <= epsilon:
                    indices.append([i, j])

        # calc data's arithmetic mean value on the circle:
        result = np.float64(0.0)
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

        self.ignored_subplots: List[List[int]] = ignore_subplots

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


# gen single subplot:
# - basic


def gen_subplot_of_vectors(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,

    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,

    data_x: np_typing.ArrayLike,
    data_y: np_typing.ArrayLike,

    stride: int,
    scale: float,

    title_data: str,

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
    ax.set_title(title_data)
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


def gen_subplot_of_streamlines(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,

    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,

    data_x: np_typing.ArrayLike,
    data_y: np_typing.ArrayLike,

    title_data: str,

    label_x: str,
    label_y: str,
) -> None:
    # The default color cycle
    color = gen_default_color_cycle()[0]

    # Plot data:
    ax.streamplot(x, y, data_x, data_y, color=color, linewidth=1, density=1.5)

    # Labels
    ax.set_title(title_data)
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


def gen_subplot_of_pseudocolor(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,

    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,

    data: np_typing.ArrayLike,

    title_data: str,

    label_x: str,
    label_y: str,
) -> None:
    # Plot data:
    pcolor = ax.pcolormesh(x, y, data, cmap='plasma', shading='auto')

    # Labels
    ax.set_title(title_data)
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


# gen single subplot:
# - avanced


def gen_subplot_of_csecs_of_2d_data_real(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,

    x: np_typing.ArrayLike,

    data: np_typing.ArrayLike,

    csec_indices: List[int],

    gen_data_csec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],

    title_data: str,

    label_x: str,
    label_data: str,

    legend_data: str,

    gen_data_csec_label_func: Callable[[np_typing.ArrayLike, int], str],
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
            label=gen_data_csec_label_func(legend_data, x, i),
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
    ax.set_title(title_data)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_data)

    # Axis limits
    min_data = np.min(data)
    max_data = np.max(data)
    ax.set_xlim(
        np.min(x),
        np.max(x)
    )
    ax.set_ylim(
        min_data - 0.02 * np.abs(min_data),
        max_data + 0.02 * np.abs(max_data)
    )

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


# gen row of subplots:


def gen_row_of_subplots_of_2d_data_real(
    fig: matplotlib.figure.Figure,
    ax: List[matplotlib.axes.Axes],

    ax_row_offset: int,

    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,

    data: np_typing.ArrayLike,

    data_xcsecs_indices: List[int],
    data_ycsecs_indices: List[int],

    gen_data_xcsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],
    gen_data_ycsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],

    title_data: str,

    label_x: str,
    label_y: str,

    label_data: str,

    legend_data: str,

    gen_data_xcsec_label_func: Callable[[np_typing.ArrayLike, int], str],
    gen_data_ycsec_label_func: Callable[[np_typing.ArrayLike, int], str],
):
    gen_subplot_of_pseudocolor(
        fig = fig,
        ax = ax[ax_row_offset + 0],

        x = x,
        y = y,

        data = data,

        title_data = title_data,

        label_x = label_x,
        label_y = label_y
    )

    gen_subplot_of_csecs_of_2d_data_real(
        fig = fig,
        ax = ax[ax_row_offset + 1],

        x = x,

        data = data,

        csec_indices = data_ycsecs_indices,

        gen_data_csec_func = gen_data_ycsec_func,

        title_data = title_data,

        label_x = label_x,
        label_data = label_data,

        legend_data = legend_data,

        gen_data_csec_label_func = gen_data_ycsec_label_func
    )

    gen_subplot_of_csecs_of_2d_data_real(
        fig = fig,
        ax = ax[ax_row_offset + 2],

        x = y,

        data = data,

        csec_indices = data_xcsecs_indices,

        gen_data_csec_func = gen_data_xcsec_func,

        title_data = title_data,

        label_x = label_y,
        label_data = label_data,

        legend_data = legend_data,

        gen_data_csec_label_func = gen_data_xcsec_label_func
    )


def gen_row_of_subplots_of_2d_data_vector(
    fig: matplotlib.figure.Figure,
    ax: List[matplotlib.axes.Axes],

    ax_row_offset: int,

    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,

    data_x: np_typing.ArrayLike,
    data_y: np_typing.ArrayLike,

    stride: int,
    scale: float,

    title_data: str,

    label_x: str,
    label_y: str,

    label_data_x: str,
    label_data_y: str,
):
    gen_subplot_of_vectors(
        fig = fig,
        ax = ax[ax_row_offset + 0],

        x = x,
        y = y,

        data_x = data_x,
        data_y = data_y,

        stride = stride,
        scale = scale,

        title_data = title_data,

        label_x = label_x,
        label_y = label_y
    )


def gen_row_of_subplots_of_2d_data_vector_current(
    fig: matplotlib.figure.Figure,
    ax: List[matplotlib.axes.Axes],

    ax_row_offset: int,

    x: np_typing.ArrayLike,
    y: np_typing.ArrayLike,

    data_x: np_typing.ArrayLike,
    data_y: np_typing.ArrayLike,

    stride: int,
    scale: float,

    title_data: str,

    label_x: str,
    label_y: str,

    label_data_x: str,
    label_data_y: str,
):
    gen_row_of_subplots_of_2d_data_vector(
        fig = fig,
        ax = ax,

        ax_row_offset = ax_row_offset + 0,

        x = x,
        y = y,

        data_x = data_x,
        data_y = data_y,

        stride = stride,
        scale = scale,

        title_data = title_data,

        label_x = label_x,
        label_y = label_y,

        label_data_x = label_data_x,
        label_data_y = label_data_y,
    )

    gen_subplot_of_streamlines(
        fig = fig,
        ax = ax[ax_row_offset + 1],

        x = x,
        y = y,

        data_x = data_x,
        data_y = data_y,

        title_data = title_data,

        label_x = label_x,
        label_y = label_y
    )


# plot grid of subplots:


def gen_grid_of_subplots_of_01_series_of_2d_data_real(
    fig: matplotlib.figure.Figure,
    ax: List[List[matplotlib.axes.Axes]],

    ax_col_offset: int,

    x: np.ndarray,
    y: np.ndarray,

    data: np.ndarray,

    data_xcsecs_indices: List[int],
    data_ycsecs_indices: List[int],

    gen_data_xcsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],
    gen_data_ycsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],

    title_data: str,

    label_x: str,
    label_y: str,

    label_data: str,

    legend_data: str,

    gen_data_xcsec_label_func: Callable[[np_typing.ArrayLike, int], str],
    gen_data_ycsec_label_func: Callable[[np_typing.ArrayLike, int], str],
) -> None:
    gen_row_of_subplots_of_2d_data_real(
        fig = fig,
        ax = ax[ax_col_offset + 0],

        ax_row_offset = 0,

        x = x,
        y = y,

        data = data,

        data_xcsecs_indices = data_xcsecs_indices,
        data_ycsecs_indices = data_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data = title_data,

        label_x = label_x,
        label_y = label_y,

        label_data = label_data,

        legend_data = legend_data,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )


def gen_grid_of_subplots_of_03_series_of_2d_data_real(
    fig: matplotlib.figure.Figure,
    ax: List[List[matplotlib.axes.Axes]],

    ax_col_offset: int,

    x: np.ndarray,
    y: np.ndarray,

    data_01: np.ndarray,
    data_02: np.ndarray,
    data_03: np.ndarray,

    data_01_xcsecs_indices: List[int],
    data_01_ycsecs_indices: List[int],

    data_02_xcsecs_indices: List[int],
    data_02_ycsecs_indices: List[int],

    data_03_xcsecs_indices: List[int],
    data_03_ycsecs_indices: List[int],

    gen_data_xcsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],
    gen_data_ycsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],

    title_data_01: str,
    title_data_02: str,
    title_data_03: str,

    label_x: str,
    label_y: str,

    label_data_01: str,
    label_data_02: str,
    label_data_03: str,

    legend_data_01: str,
    legend_data_02: str,
    legend_data_03: str,

    gen_data_xcsec_label_func: Callable[[np_typing.ArrayLike, int], str],
    gen_data_ycsec_label_func: Callable[[np_typing.ArrayLike, int], str],
) -> None:
    gen_grid_of_subplots_of_01_series_of_2d_data_real(
        fig = fig,
        ax = ax,

        ax_col_offset = ax_col_offset + 0,

        x = x,
        y = y,

        data = data_01,

        data_xcsecs_indices = data_01_xcsecs_indices,
        data_ycsecs_indices = data_01_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data = title_data_01,

        label_x = label_x,
        label_y = label_y,

        label_data = label_data_01,

        legend_data = legend_data_01,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )

    gen_grid_of_subplots_of_01_series_of_2d_data_real(
        fig = fig,
        ax = ax,

        ax_col_offset = ax_col_offset + 1,

        x = x,
        y = y,

        data = data_02,

        data_xcsecs_indices = data_02_xcsecs_indices,
        data_ycsecs_indices = data_02_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data = title_data_02,

        label_x = label_x,
        label_y = label_y,

        label_data = label_data_02,

        legend_data = legend_data_02,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )

    gen_grid_of_subplots_of_01_series_of_2d_data_real(
        fig = fig,
        ax = ax,

        ax_col_offset = ax_col_offset + 2,

        x = x,
        y = y,

        data = data_03,

        data_xcsecs_indices = data_03_xcsecs_indices,
        data_ycsecs_indices = data_03_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data = title_data_03,

        label_x = label_x,
        label_y = label_y,

        label_data = label_data_03,

        legend_data = legend_data_03,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )


def gen_grid_of_subplots_of_01_series_of_2d_data_vector_current(
    fig: matplotlib.figure.Figure,
    ax: List[List[matplotlib.axes.Axes]],

    ax_col_offset: int,

    x: np.ndarray,
    y: np.ndarray,

    data_norm: np.ndarray,
    data_x: np.ndarray,
    data_y: np.ndarray,

    stride: int,
    scale: float,

    data_norm_xcsecs_indices: List[int],
    data_norm_ycsecs_indices: List[int],

    data_x_xcsecs_indices: List[int],
    data_x_ycsecs_indices: List[int],

    data_y_xcsecs_indices: List[int],
    data_y_ycsecs_indices: List[int],

    gen_data_xcsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],
    gen_data_ycsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],

    title_data: str,
    title_data_norm: str,
    title_data_x: str,
    title_data_y: str,

    label_x: str,
    label_y: str,

    label_data_norm: str,
    label_data_x: str,
    label_data_y: str,

    legend_data_norm: str,
    legend_data_x: str,
    legend_data_y: str,

    gen_data_xcsec_label_func: Callable[[np_typing.ArrayLike, int], str],
    gen_data_ycsec_label_func: Callable[[np_typing.ArrayLike, int], str],
) -> None:
    gen_row_of_subplots_of_2d_data_vector_current(
        fig = fig,
        ax = ax[ax_col_offset + 0],

        ax_row_offset = 0,

        x = x,
        y = y,

        data_x = data_x,
        data_y = data_y,

        stride = stride,
        scale = scale,

        title_data = title_data,

        label_x = label_x,
        label_y = label_y,

        label_data_x = '',
        label_data_y = '',
    )

    gen_grid_of_subplots_of_03_series_of_2d_data_real(
        fig = fig,
        ax = ax,

        ax_col_offset = ax_col_offset + 1,

        x = x,
        y = y,

        data_01 = data_norm,
        data_02 = data_x,
        data_03 = data_y,

        data_01_xcsecs_indices = data_norm_xcsecs_indices,
        data_01_ycsecs_indices = data_norm_ycsecs_indices,

        data_02_xcsecs_indices = data_x_xcsecs_indices,
        data_02_ycsecs_indices = data_x_ycsecs_indices,

        data_03_xcsecs_indices = data_y_xcsecs_indices,
        data_03_ycsecs_indices = data_y_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data_01 = title_data_norm,
        title_data_02 = title_data_x,
        title_data_03 = title_data_y,

        label_x = label_x,
        label_y = label_y,

        label_data_01 = label_data_norm,
        label_data_02 = label_data_x,
        label_data_03 = label_data_y,

        legend_data_01 = legend_data_norm,
        legend_data_02 = legend_data_x,
        legend_data_03 = legend_data_y,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )


def gen_grid_of_subplots_of_03_series_of_2d_data_vector_current(
    fig: matplotlib.figure.Figure,
    ax: List[List[matplotlib.axes.Axes]],

    ax_col_offset: int,

    x: np.ndarray,
    y: np.ndarray,

    data_01_norm: np.ndarray,
    data_01_x: np.ndarray,
    data_01_y: np.ndarray,

    data_02_norm: np.ndarray,
    data_02_x: np.ndarray,
    data_02_y: np.ndarray,

    data_03_norm: np.ndarray,
    data_03_x: np.ndarray,
    data_03_y: np.ndarray,

    stride: int,
    scale: float,

    data_norm_xcsecs_indices: List[int],
    data_norm_ycsecs_indices: List[int],

    data_x_xcsecs_indices: List[int],
    data_x_ycsecs_indices: List[int],

    data_y_xcsecs_indices: List[int],
    data_y_ycsecs_indices: List[int],

    gen_data_xcsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],
    gen_data_ycsec_func: Callable[[np_typing.ArrayLike, int], np_typing.ArrayLike],

    title_data_01: str,
    title_data_01_norm: str,
    title_data_01_x: str,
    title_data_01_y: str,

    title_data_02: str,
    title_data_02_norm: str,
    title_data_02_x: str,
    title_data_02_y: str,

    title_data_03: str,
    title_data_03_norm: str,
    title_data_03_x: str,
    title_data_03_y: str,

    label_x: str,
    label_y: str,

    label_data_01_norm: str,
    label_data_01_x: str,
    label_data_01_y: str,

    label_data_02_norm: str,
    label_data_02_x: str,
    label_data_02_y: str,

    label_data_03_norm: str,
    label_data_03_x: str,
    label_data_03_y: str,

    legend_data_01_norm: str,
    legend_data_01_x: str,
    legend_data_01_y: str,

    legend_data_02_norm: str,
    legend_data_02_x: str,
    legend_data_02_y: str,

    legend_data_03_norm: str,
    legend_data_03_x: str,
    legend_data_03_y: str,

    gen_data_xcsec_label_func: Callable[[np_typing.ArrayLike, int], str],
    gen_data_ycsec_label_func: Callable[[np_typing.ArrayLike, int], str],

) -> None:
    gen_grid_of_subplots_of_01_series_of_2d_data_vector_current(
        fig = fig,
        ax = ax,

        ax_col_offset = ax_col_offset + 0,

        x = x,
        y = y,

        data_norm = data_01_norm,
        data_x = data_01_x,
        data_y = data_01_y,

        stride = stride,
        scale = scale,

        data_norm_xcsecs_indices = data_norm_xcsecs_indices,
        data_norm_ycsecs_indices = data_norm_ycsecs_indices,

        data_x_xcsecs_indices = data_x_xcsecs_indices,
        data_x_ycsecs_indices = data_x_ycsecs_indices,

        data_y_xcsecs_indices = data_y_xcsecs_indices,
        data_y_ycsecs_indices = data_y_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data = title_data_01,
        title_data_norm = title_data_01_norm,
        title_data_x = title_data_01_x,
        title_data_y = title_data_01_y,

        label_x = label_x,
        label_y = label_y,

        label_data_norm = label_data_01_norm,
        label_data_x = label_data_01_x,
        label_data_y = label_data_01_y,

        legend_data_norm = legend_data_01_norm,
        legend_data_x = legend_data_01_x,
        legend_data_y = legend_data_01_y,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )

    gen_grid_of_subplots_of_01_series_of_2d_data_vector_current(
        fig = fig,
        ax = ax,

        ax_col_offset = ax_col_offset + 4,

        x = x,
        y = y,

        data_norm = data_02_norm,
        data_x = data_02_x,
        data_y = data_02_y,

        stride = stride,
        scale = scale,

        data_norm_xcsecs_indices = data_norm_xcsecs_indices,
        data_norm_ycsecs_indices = data_norm_ycsecs_indices,

        data_x_xcsecs_indices = data_x_xcsecs_indices,
        data_x_ycsecs_indices = data_x_ycsecs_indices,

        data_y_xcsecs_indices = data_y_xcsecs_indices,
        data_y_ycsecs_indices = data_y_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data = title_data_02,
        title_data_norm = title_data_02_norm,
        title_data_x = title_data_02_x,
        title_data_y = title_data_02_y,

        label_x = label_x,
        label_y = label_y,

        label_data_norm = label_data_02_norm,
        label_data_x = label_data_02_x,
        label_data_y = label_data_02_y,

        legend_data_norm = legend_data_02_norm,
        legend_data_x = legend_data_02_x,
        legend_data_y = legend_data_02_y,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )

    gen_grid_of_subplots_of_01_series_of_2d_data_vector_current(
        fig = fig,
        ax = ax,

        ax_col_offset = ax_col_offset + 8,

        x = x,
        y = y,

        data_norm = data_03_norm,
        data_x = data_03_x,
        data_y = data_03_y,

        stride = stride,
        scale = scale,

        data_norm_xcsecs_indices = data_norm_xcsecs_indices,
        data_norm_ycsecs_indices = data_norm_ycsecs_indices,

        data_x_xcsecs_indices = data_x_xcsecs_indices,
        data_x_ycsecs_indices = data_x_ycsecs_indices,

        data_y_xcsecs_indices = data_y_xcsecs_indices,
        data_y_ycsecs_indices = data_y_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec_func,
        gen_data_ycsec_func = gen_data_ycsec_func,

        title_data = title_data_03,
        title_data_norm = title_data_03_norm,
        title_data_x = title_data_03_x,
        title_data_y = title_data_03_y,

        label_x = label_x,
        label_y = label_y,

        label_data_norm = label_data_03_norm,
        label_data_x = label_data_03_x,
        label_data_y = label_data_03_y,

        legend_data_norm = legend_data_03_norm,
        legend_data_x = legend_data_03_x,
        legend_data_y = legend_data_03_y,

        gen_data_xcsec_label_func = gen_data_xcsec_label_func,
        gen_data_ycsec_label_func = gen_data_ycsec_label_func,
    )


# plot systen properties:


def __plot_01_normal_density(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    rho_tot: np.ndarray,
    rho_a: np.ndarray,
    rho_b: np.ndarray,

    title_rho_tot: str,
    title_rho_a: str,
    title_rho_b: str,

    label_x: str,
    label_y: str,

    label_rho_tot: str,
    label_rho_a: str,
    label_rho_b: str,

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

    data_xcsecs_indices = gen_csecs_indices(nx/2)
    data_ycsecs_indices = gen_csecs_indices(ny/2)

    gen_data_xcsec = lambda data, x: data[x, :]
    gen_data_ycsec = lambda data, y: data[:, y]

    gen_data_xcsec_label = lambda legend, x, i: f"{legend}(x = {x[i]}, y)"
    gen_data_ycsec_label = lambda legend, y, i: f"{legend}(x, y = {y[i]})"

    gen_grid_of_subplots_of_03_series_of_2d_data_real(
        fig = p.fig,
        ax = p.ax,

        ax_col_offset = 0,

        x = x,
        y = y,

        data_01 = rho_tot,
        data_02 = rho_a,
        data_03 = rho_b,

        data_01_xcsecs_indices = data_xcsecs_indices,
        data_01_ycsecs_indices = data_ycsecs_indices,

        data_02_xcsecs_indices = data_xcsecs_indices,
        data_02_ycsecs_indices = data_ycsecs_indices,

        data_03_xcsecs_indices = data_xcsecs_indices,
        data_03_ycsecs_indices = data_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec,
        gen_data_ycsec_func = gen_data_ycsec,

        title_data_01 = title_rho_tot,
        title_data_02 = title_rho_a,
        title_data_03 = title_rho_b,

        label_x = label_x,
        label_y = label_y,

        label_data_01 = label_rho_tot,
        label_data_02 = label_rho_a,
        label_data_03 = label_rho_b,

        legend_data_01 = legend_rho_tot,
        legend_data_02 = legend_rho_a,
        legend_data_03 = legend_rho_b,

        gen_data_xcsec_label_func = gen_data_xcsec_label,
        gen_data_ycsec_label_func = gen_data_ycsec_label,
    )

    p.fig.tight_layout()
    p.fig.savefig(fname = out_file, dpi = dpi)


def __plot_02_pairing_gap_function(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    delta_norm: np.ndarray,

    title_delta_norm: str,

    label_x: str,
    label_y: str,

    label_delta_norm: str,

    legend_delta_norm: str,

) -> None:
    p = Plot(
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        cols_w_ratios = [1.11, 1, 1],
        rows_h_ratios = [1],
    )

    data_xcsecs_indices = gen_csecs_indices(nx/2)
    data_ycsecs_indices = gen_csecs_indices(ny/2)

    gen_data_xcsec = lambda data, x: data[x, :]
    gen_data_ycsec = lambda data, y: data[:, y]

    gen_data_xcsec_label = lambda legend, x, i: f"{legend}(x = {x[i]}, y)"
    gen_data_ycsec_label = lambda legend, y, i: f"{legend}(x, y = {y[i]})"

    gen_grid_of_subplots_of_01_series_of_2d_data_real(
        fig = p.fig,
        ax = p.ax,

        ax_col_offset = 0,

        x = x,
        y = y,

        data = delta_norm,

        data_xcsecs_indices = data_xcsecs_indices,
        data_ycsecs_indices = data_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec,
        gen_data_ycsec_func = gen_data_ycsec,

        title_data = title_delta_norm,

        label_x = label_x,
        label_y = label_y,

        label_data = label_delta_norm,

        legend_data = legend_delta_norm,

        gen_data_xcsec_label_func = gen_data_xcsec_label,
        gen_data_ycsec_label_func = gen_data_ycsec_label,
    )

    # save plot:

    p.fig.tight_layout()
    p.fig.savefig(fname = out_file, dpi = dpi)


def __plot_03_current_density(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

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

    label_x: str,
    label_y: str,

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

    data_norm_xcsecs_indices = gen_csecs_indices(nx/2)
    data_norm_ycsecs_indices = gen_csecs_indices(ny/2)

    data_x_xcsecs_indices = gen_csecs_indices(nx/2, [0.75, 0.875, 0.95])
    data_x_ycsecs_indices = gen_csecs_indices(ny/2, [0.75, 0.875, 0.95], [1.05, 1.125, 1.25])

    data_y_xcsecs_indices = gen_csecs_indices(nx/2, [0.75, 0.875, 0.95], [1.05, 1.125, 1.25])
    data_y_ycsecs_indices = gen_csecs_indices(ny/2, [0.75, 0.875, 0.95])

    gen_data_xcsec = lambda data, x: data[:, x]
    gen_data_ycsec = lambda data, y: data[y, :]

    gen_data_xcsec_label = lambda legend, x, i: f"{legend}(x = {x[i]}, y)"
    gen_data_ycsec_label = lambda legend, y, i: f"{legend}(x, y = {y[i]})"

    vector_stride = 3
    vector_scale = 0.0005

    gen_grid_of_subplots_of_03_series_of_2d_data_vector_current(
        fig = p.fig,
        ax = p.ax,

        ax_col_offset = 0,

        x = x,
        y = y,

        data_01_norm = j_tot_norm,
        data_01_x = j_tot_x,
        data_01_y = j_tot_y,

        data_02_norm = j_a_norm,
        data_02_x = j_a_x,
        data_02_y = j_a_y,

        data_03_norm = j_b_norm,
        data_03_x = j_b_x,
        data_03_y = j_b_y,

        stride = vector_stride,
        scale = vector_scale,

        data_norm_xcsecs_indices = data_norm_xcsecs_indices,
        data_norm_ycsecs_indices = data_norm_ycsecs_indices,

        data_x_xcsecs_indices = data_x_xcsecs_indices,
        data_x_ycsecs_indices = data_x_ycsecs_indices,

        data_y_xcsecs_indices = data_y_xcsecs_indices,
        data_y_ycsecs_indices = data_y_ycsecs_indices,

        gen_data_xcsec_func = gen_data_xcsec,
        gen_data_ycsec_func = gen_data_ycsec,

        title_data_01 = title_j_tot,
        title_data_01_norm = title_j_tot_norm,
        title_data_01_x = title_j_tot_x,
        title_data_01_y = title_j_tot_y,

        title_data_02 = title_j_a,
        title_data_02_norm = title_j_a_norm,
        title_data_02_x = title_j_a_x,
        title_data_02_y = title_j_a_y,

        title_data_03 = title_j_b,
        title_data_03_norm = title_j_b_norm,
        title_data_03_x = title_j_b_x,
        title_data_03_y = title_j_b_y,

        label_x = label_x,
        label_y = label_y,

        label_data_01_norm = label_j_tot_norm,
        label_data_01_x = label_j_tot_x,
        label_data_01_y = label_j_tot_y,

        label_data_02_norm = label_j_a_norm,
        label_data_02_x = label_j_a_x,
        label_data_02_y = label_j_a_y,

        label_data_03_norm = label_j_b_norm,
        label_data_03_x = label_j_b_x,
        label_data_03_y = label_j_b_y,

        legend_data_01_norm = legend_j_tot_norm,
        legend_data_01_x = legend_j_tot_x,
        legend_data_01_y = legend_j_tot_y,

        legend_data_02_norm = legend_j_a_norm,
        legend_data_02_x = legend_j_a_x,
        legend_data_02_y = legend_j_a_y,

        legend_data_03_norm = legend_j_b_norm,
        legend_data_03_x = legend_j_b_x,
        legend_data_03_y = legend_j_b_y,

        gen_data_xcsec_label_func = gen_data_xcsec_label,
        gen_data_ycsec_label_func = gen_data_ycsec_label,
    )

    # save plot:

    p.fig.tight_layout()
    p.fig.savefig(fname = out_file, dpi = dpi)


# plot systen properties:


def plot_01_normal_density(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedWData
) -> None:
    __plot_01_normal_density(
        out_file = out_dir + "/plot_01_normal_densit.png",
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

        label_x = 'x',
        label_y = 'y',

        label_rho_tot = 'rho_tot',
        label_rho_a = 'rho_a',
        label_rho_b = 'rho_b',

        legend_rho_tot = 'rho_tot',
        legend_rho_a = 'rho_a',
        legend_rho_b = 'rho_b',
    )

    __plot_01_normal_density(
        out_file = out_dir + "/plot_01_normal_densit_normalized.png",
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

        label_x = 'x',
        label_y = 'y',

        label_rho_tot = 'rho_total / rho_total_0',
        label_rho_a = 'rho_a / rho_a_0',
        label_rho_b = 'rho_b / rho_b_0',

        legend_rho_tot = 'rho_total / rho_total_0 ',
        legend_rho_a = 'rho_a / rho_a_0 ',
        legend_rho_b = 'rho_b / rho_b_0 ',
    )


def plot_02_pairing_gap_function(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedWData
) -> None:
    __plot_02_pairing_gap_function(
        out_file = out_dir + "/plot_02_pairing_gap_function.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        delta_norm = parsed.delta_norm,

        title_delta_norm = '|Δ|',

        label_x = 'x',
        label_y = 'y',

        label_delta_norm = '|Δ|',

        legend_delta_norm = '|Δ|',
    )

    __plot_02_pairing_gap_function(
        out_file = out_dir + "/plot_02_pairing_gap_function_normalized.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        delta_norm = parsed.delta_norm / parsed.delta_norm_0,

        title_delta_norm = '|Δ| / |Δ_0|',

        label_x = 'x',
        label_y = 'y',

        label_delta_norm = '|Δ| / |Δ_0|',

        legend_delta_norm = '|Δ| / |Δ_0| ',
    )


def plot_03_current_density(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedWData
) -> None:
    __plot_03_current_density(
        out_file = out_dir + "/plot_03_current_density.png",
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

        label_x = 'x',
        label_y = 'y',

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

    plot_01_normal_density(
        out_dir = out_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_02_pairing_gap_function(
        out_dir = out_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_03_current_density(
        out_dir = out_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
