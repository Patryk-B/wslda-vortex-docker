#!/usr/bin/env python3

from enum import Enum
from matplotlib import ticker
from scipy.interpolate import interp1d
from typing import Any, Union, Tuple, List, Set, Dict, DefaultDict, Iterator, Callable
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
from collections import defaultdict, namedtuple
from itertools import groupby
import traceback
import logging
import paramiko
import tempfile
import re

import json

import os
import subprocess
import time
import shutil

import subprocess
import os
import time

#! ---- . ---- ---- ---- ---- . ----
#! helpers:
#! ---- . ---- ---- ---- ---- . ----


class Axis(int, Enum):
    X = 0
    Y = 1
    Z = 2


class Component(int, Enum):
    X = 0
    Y = 1
    Z = 2


class IJRPoint:
    def __init__(
        self,
        i: int,
        j: int,
        x: float,
        y: float,
        r: float,
    ):
        self.i: int = i
        self.j: int = j
        self.x: float = x
        self.y: float = y
        self.r: float = r

    def __str__(self):
        return f"(i: {self.i:5}, j: {self.j:5}, x: {self.x:10}, y: {self.y:10}, r: {self.r:10})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return (self.r, self.i, self.j) < (other.r, other.i, other.j)

    def __le__(self, other):
        return (self.r, self.i, self.j) <= (other.r, other.i, other.j)

    def __eq__(self, other):
        return (self.r, self.i, self.j) == (other.r, other.i, other.j)

    def __ne__(self, other):
        return (self.r, self.i, self.j) != (other.r, other.i, other.j)

    def __gt__(self, other):
        return (self.r, self.i, self.j) > (other.r, other.i, other.j)

    def __ge__(self, other):
        return (self.r, self.i, self.j) >= (other.r, other.i, other.j)


class IJRGrid:
    def __init__(self, x: List[float], y: List[float]):
        # Initialize grid
        self.__x = x
        self.__y = y
        self.__grid = np.array([])
        for i in range(len(self.__x)):
            for j in range(len(self.__y)):
                x = self.__x[i]
                y = self.__y[j]
                r = np.sqrt(x ** 2 + y ** 2)
                self.__grid = np.append(
                    self.__grid,
                    IJRPoint(i, j, x, y, r)
                )

        # Sort grid by 'r' attribute
        self.__grid = np.sort(self.__grid)

        # Group by 'r'
        self.__grid_dict = defaultdict(lambda: np.array([], dtype=object))
        for point in self.__grid:
            self.__grid_dict[point.r] = np.append(self.__grid_dict[point.r], point)

        # Convert keys (r values) to a numpy array
        self.__r = np.array(list(self.__grid_dict.keys()))

        # Convert values (lists of IJRPoint) to a numpy array
        self.__ij = np.array(list(self.__grid_dict.values()), dtype=object)

    # Length based on unique 'r' values
    def __len__(self):
        return len(self.__r)

    # Iterator over all groups of points
    def __iter__(self) -> Iterator[Tuple[float, np.array]]:
        for r, points_grouped_by_r in zip(self.__r, self.__ij):
            yield (r, points_grouped_by_r)

    def __getitem__(self, i: int) -> Tuple[float, np.array]:
        if not isinstance(i, int):
            raise TypeError("Index must be an integer")
        if i < 0 or i >= len(self):
            raise IndexError("Index out of range")
        return (self.__r[i], self.__ij[i])

    def get_points_by_radius(
        self,
        radius: float,
        radius_precision: float = 0.01
    ):
        i = np.argmin(np.abs(self.__r - radius))
        r = self.__r[i]

        if np.abs(r - radius) <= radius_precision:
            return self.__ij[i]
        else:
            return None


class ParsedData(object):
    def __init__(
        self,
        data_dir,
    ):
        #! ---- . ---- ---- ---- ---- . ----
        #! load
        #! ---- . ---- ---- ---- ---- . ----

        wdata = self.__load_wdata(data_dir)
        input = self.__load_input(data_dir)
        iteration = -1

        #? ---- . ---- ---- ---- ---- . ----
        #? grid:
        #? ---- . ---- ---- ---- ---- . ----

        self.nx: int = wdata.Nxyz[Axis.X]
        self.ny: int = wdata.Nxyz[Axis.Y]

        self.nx_half: int = int(self.nx / 2)
        self.ny_half: int = int(self.ny / 2)

        self.nx_fourth: int = int(self.nx_half / 2)
        self.ny_fourth: int = int(self.ny_half / 2)

        self.nx_eighth: int = int(self.nx_fourth / 2)
        self.ny_eighth: int = int(self.ny_fourth / 2)

        self.nx_sixteenth: int = int(self.nx_eighth / 2)
        self.ny_sixteenth: int = int(self.ny_eighth / 2)

        self.dx: float = wdata.dxyz[Axis.X]
        self.dy: float = wdata.dxyz[Axis.Y]

        self.x: np.ndarray = wdata.xyz[Axis.X]
        self.y: np.ndarray = wdata.xyz[Axis.Y]

        self.x_flat: np.ndarray = self.x.flatten()
        self.y_flat: np.ndarray = self.y.flatten()

        self.ijrGrid = IJRGrid(
            self.x_flat,
            self.y_flat
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? radius:
        #? - center of the vortex core lies at the center of the grid
        #?   r_v = x[nx * 1/2]
        #?       = y[ny * 1/2]
        #?       = 0
        #?
        #? - vortex bulk spans a circle with r = r_0
        #?   r_0 = abs( x[nx * 1/2 - nx * 1/4] ) = abs( x[nx * 1/4] )
        #?       = abs( x[nx * 1/2 + nx * 1/4] ) = abs( x[nx * 3/4] )
        #?       = abs( y[ny * 1/2 - ny * 1/4] ) = abs( y[ny * 1/4] )
        #?       = abs( y[ny * 1/2 + ny * 1/4] ) = abs( y[ny * 3/4] )
        #?       = 20
        #? ---- . ---- ---- ---- ---- . ----

        self.x_v_index = self.nx_half
        self.y_v_index = self.ny_half

        self.x_0_index = self.nx_half + self.nx_fourth
        self.y_0_index = self.ny_half + self.ny_fourth

        self.x_v = self.x_flat[self.x_v_index]
        self.y_v = self.y_flat[self.y_v_index]

        self.x_0 = self.x_flat[self.x_0_index]
        self.y_0 = self.y_flat[self.y_0_index]

        self.r_v = self.x_v
        self.r_0 = self.x_0
        self.r_precision = 0.01

        #? ---- . ---- ---- ---- ---- . ----
        #? rho (x,y) -> Real
        #? - normal density (x,y)
        #?
        #? - NOTE:
        #?   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #?     - wdata.rho_a[iteration][ nx/2, :    ] === rho_a( x = nx/2, y        )
        #?     - wdata.rho_a[iteration][ :   , ny/2 ] === rho_a( x       , y = ny/2 )
        #? ---- . ---- ---- ---- ---- . ----

        self.rho_a: np.memmap = wdata.rho_a[iteration]
        self.rho_b: np.memmap = wdata.rho_b[iteration]
        self.rho_tot: np.memmap = self.rho_a + self.rho_b

        # ---- . ---- ---- ---- ---- . ----
        # rho_v = <rho(r = r_v = 0)>
        # - vortex normal density
        # - normal density at the center of the vortex core
        # ---- . ---- ---- ---- ---- . ----

        self.rho_a_v: np.float64 = self.rho_a[self.nx_half, self.ny_half]
        self.rho_b_v: np.float64 = self.rho_b[self.nx_half, self.ny_half]
        self.rho_tot_v: np.float64 = self.rho_tot[self.nx_half, self.ny_half]

        # ---- . ---- ---- ---- ---- . ----
        # rho_0 = <rho(r = r_0)>
        # - bulk normal density
        # - normal density far from the center of the vortex core
        # ---- . ---- ---- ---- ---- . ----

        self.rho_a_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.rho_a,
            self.r_0,
            self.r_precision
        )
        self.rho_b_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.rho_b,
            self.r_0,
            self.r_precision
        )
        self.rho_tot_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.rho_tot,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? delta (x,y) -> Conplex
        #? - pairing gap function (x,y)
        #?
        #? - NOTE:
        #?   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #?     - wdata.delta[iteration][ nx/2, :    ] === delta( x = nx/2, y        )
        #?     - wdata.delta[iteration][ :   , ny/2 ] === delta( x       , y = ny/2 )
        #? ---- . ---- ---- ---- ---- . ----

        self.delta: np.memmap = wdata.delta[iteration]
        self.delta_norm: np.ndarray = np.abs(self.delta)

        # ---- . ---- ---- ---- ---- . ----
        # delta_norm_0 = <delta_norm(r = r_0)>
        # - bulk pairing gap function
        # - pairing gap function far from the center of the vortex core
        # ---- . ---- ---- ---- ---- . ----

        self.delta_norm_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.delta_norm,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? j (x,y) -> Vector
        #? - current density (x,y)
        #?
        #? - WARNING:
        #?   - j is transposed !!!
        #?
        #? - NOTE:
        #?   - according to https://gitlab.fizyka.pw.edu.pl/wtools/wdata/-/wikis/Examples/Python-examples#cross-section-of-velocity-field-for-quantum-vortex:
        #?     - wdata.j_a[iteration][Component.X][ nx/2, :    ] === j_a_x( x = nx/2, y        )
        #?     - wdata.j_a[iteration][Component.X][ :   , ny/2 ] === j_a_x( x       , y = ny/2 )
        #? ---- . ---- ---- ---- ---- . ----

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

        # # ---- . ---- ---- ---- ---- . ----
        # # j_max = max( abs( j(r) ) )
        # # - max current density
        # # ---- . ---- ---- ---- ---- . ----

        # self.j_a_norm_max: np.float64 = np.max(np.abs(self.j_a_norm))
        # self.j_a_x_max: np.float64 = np.max(np.abs(self.j_a_x))
        # self.j_a_y_max: np.float64 = np.max(np.abs(self.j_a_y))

        # self.j_b_norm_max: np.float64 = np.max(np.abs(self.j_b_norm))
        # self.j_b_x_max: np.float64 = np.max(np.abs(self.j_b_x))
        # self.j_b_y_max: np.float64 = np.max(np.abs(self.j_b_y))

        # self.j_tot_norm_max: np.float64 = np.max(np.abs(self.j_tot_norm))
        # self.j_tot_x_max: np.float64 = np.max(np.abs(self.j_tot_x))
        # self.j_tot_y_max: np.float64 = np.max(np.abs(self.j_tot_y))

        #? ---- . ---- ---- ---- ---- . ----
        #? nu (x,y) -> Complex
        #? -
        #? ---- . ---- ---- ---- ---- . ----

        self.nu: np.memmap = wdata.nu[iteration]
        self.nu_norm: np.ndarray = np.abs(self.nu)

        # ---- . ---- ---- ---- ---- . ----
        # nu_norm_0 = <nu_norm(r = r_0)>
        # -
        # ---- . ---- ---- ---- ---- . ----

        self.nu_norm_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.nu_norm,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? tau (x,y) -> Real
        #? -
        #? ---- . ---- ---- ---- ---- . ----

        self.tau_a: np.memmap = wdata.tau_a[iteration]
        self.tau_b: np.memmap = wdata.tau_b[iteration]
        self.tau_tot: np.memmap = self.tau_a + self.tau_b

        # ---- . ---- ---- ---- ---- . ----
        # tau_0 = <tau(r = r_0)>
        # -
        # ---- . ---- ---- ---- ---- . ----

        self.tau_a_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.tau_a,
            self.r_0,
            self.r_precision
        )
        self.tau_b_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.tau_b,
            self.r_0,
            self.r_precision
        )
        self.tau_tot_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.tau_tot,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? V (x,y) -> Real
        #? -
        #? ---- . ---- ---- ---- ---- . ----

        self.V_a: np.memmap = wdata.V_a[iteration]
        self.V_b: np.memmap = wdata.V_b[iteration]
        self.V_tot: np.memmap = self.V_a + self.V_b

        # ---- . ---- ---- ---- ---- . ----
        # V_0 = <V(r = r_0)>
        # -
        # ---- . ---- ---- ---- ---- . ----

        self.V_a_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.V_a,
            self.r_0,
            self.r_precision
        )
        self.V_b_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.V_b,
            self.r_0,
            self.r_precision
        )
        self.V_tot_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.V_tot,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? V_ext (x,y) -> Real
        #? -
        #? ---- . ---- ---- ---- ---- . ----

        self.V_ext_a: np.memmap = wdata.V_ext_a[iteration]
        self.V_ext_b: np.memmap = wdata.V_ext_b[iteration]
        self.V_ext_tot: np.memmap = self.V_ext_a + self.V_ext_b

        # ---- . ---- ---- ---- ---- . ----
        # V_ext_0 = <V_ext(r = r_0)>
        # -
        # ---- . ---- ---- ---- ---- . ----

        self.V_ext_a_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.V_ext_a,
            self.r_0,
            self.r_precision
        )
        self.V_ext_b_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.V_ext_b,
            self.r_0,
            self.r_precision
        )
        self.V_ext_tot_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.V_ext_tot,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? delta_ext (x,y) -> Complex
        #? -
        #? ---- . ---- ---- ---- ---- . ----

        self.delta_ext: np.memmap = wdata.delta_ext[iteration]
        self.delta_ext_norm: np.ndarray = np.abs(self.delta_ext)

        # ---- . ---- ---- ---- ---- . ----
        # delta_ext_norm_0 = <delta_ext_norm(r = r_0)>
        # -
        # ---- . ---- ---- ---- ---- . ----

        self.delta_ext_norm_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.delta_ext_norm,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? velocity_ext (x,y) -> Vector
        #? -
        #? ---- . ---- ---- ---- ---- . ----

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

        #? ---- . ---- ---- ---- ---- . ----
        #? alpha (x,y) -> Real
        #? -
        #? ---- . ---- ---- ---- ---- . ----

        self.alpha_a: np.memmap = wdata.alpha_a[iteration]
        self.alpha_b: np.memmap = wdata.alpha_b[iteration]
        self.alpha_tot: np.memmap = self.alpha_a + self.alpha_b

        # ---- . ---- ---- ---- ---- . ----
        # alpha_0 = <alpha(r_0)>
        # -
        # ---- . ---- ---- ---- ---- . ----

        self.alpha_a_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.alpha_a,
            self.r_0,
            self.r_precision
        )
        self.alpha_b_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.alpha_b,
            self.r_0,
            self.r_precision
        )
        self.alpha_tot_0: np.float64 = self.__calc_arithmetic_mean_over_circle(
            self.alpha_tot,
            self.r_0,
            self.r_precision
        )

        #? ---- . ---- ---- ---- ---- . ----
        #? A (x,y) -> Vector
        #? -
        #?
        #? - WARNING:
        #?   - j is transposed !!!
        #? ---- . ---- ---- ---- ---- . ----

        self.A_a_x: np.memmap = wdata.A_a[iteration][Component.X].T
        self.A_a_y: np.memmap = wdata.A_a[iteration][Component.Y].T
        self.A_a: np.ndarray = np.column_stack((self.A_a_x, self.A_a_y))
        self.A_a_norm: np.ndarray = np.sqrt(self.A_a_x ** 2 + self.A_a_y ** 2)

        self.A_b_x: np.memmap = wdata.A_b[iteration][Component.X].T
        self.A_b_y: np.memmap = wdata.A_b[iteration][Component.Y].T
        self.A_b: np.ndarray = np.column_stack((self.A_b_x, self.A_b_y))
        self.A_b_norm: np.ndarray = np.sqrt(self.A_b_x ** 2 + self.A_b_y ** 2)

        self.A_tot_x: np.memmap = self.A_a_x + self.A_b_x
        self.A_tot_y: np.memmap = self.A_a_y + self.A_b_y
        self.A_tot: np.ndarray = np.column_stack((self.A_tot_x, self.A_tot_y))
        self.A_tot_norm: np.ndarray = np.sqrt(self.A_tot_x ** 2 + self.A_tot_y ** 2)

        #! ---- . ---- ---- ---- ---- . ----
        #! calculated:
        #
        #! TODO:
        #! - plot following as a paraneter of unctions of r
        #! ---- . ---- ---- ---- ---- . ----

        # ---- . ---- ---- ---- ---- . ----
        # mass_star
        # - effective mass
        # ---- . ---- ---- ---- ---- . ----

        self.mass_star: np.float64 = np.float64(1.0 / self.alpha_a_0)

        # ---- . ---- ---- ---- ---- . ----
        # k_F
        # - fermi momentum
        # ---- . ---- ---- ---- ---- . ----

        self.k_F: np.float64 = np.cbrt(3.0 * (np.pi ** 2) * self.rho_tot_0)

        # ---- . ---- ---- ---- ---- . ----
        # epislon_F
        # - fermi energy
        # ---- . ---- ---- ---- ---- . ----

        self.E_F: np.float64 = np.float64((self.k_F ** 2) / 2.0)

        # ---- . ---- ---- ---- ---- . ----
        # epislon_F_star
        # - effective fermi energy
        # ---- . ---- ---- ---- ---- . ----

        self.E_F_star: np.float64 = np.float64(self.alpha_a_0 * self.E_F)

        # ---- . ---- ---- ---- ---- . ----
        # T
        # - temperature
        # ---- . ---- ---- ---- ---- . ----

        self.T: np.float64 = np.float64(input['temperature'])

        # ---- . ---- ---- ---- ---- . ----
        # T_F
        # - fermi temperature
        # ---- . ---- ---- ---- ---- . ----

        self.T_F: np.float64 = self.E_F

        # ---- . ---- ---- ---- ---- . ----
        # T_c
        # - fermi temperature
        # ---- . ---- ---- ---- ---- . ----

        self.T_c = None
        if self.T == 0.0:
            self.T_c = np.float64(1.764 * self.delta_norm_0 * self.T_F / self.E_F_star)

        # ---- . ---- ---- ---- ---- . ----
        # l_c
        # - coherence length
        # ---- . ---- ---- ---- ---- . ----

        self.l_c: np.float64 = np.float64((2.0 * self.E_F_star) / (np.pi * self.k_F * self.delta_norm_0))

        # ---- . ---- ---- ---- ---- . ----
        # a_s
        # - scattering length
        # ---- . ---- ---- ---- ---- . ----

        self.a_s: np.float64 = np.float64(input['sclgth'])

        # ---- . ---- ---- ---- ---- . ----
        # lambda
        # - density-dependent coupling constant
        # ---- . ---- ---- ---- ---- . ----

        self.lmb: np.float64 = np.abs(self.a_s * self.k_F)


    def __json__(
        self
    ) -> object:
        return {
            # ---- . ---- ---- ---- ---- . ----
            # grid
            # ---- . ---- ---- ---- ---- . ----
            'grid': {
                'nx': self.nx,
                'ny': self.ny,

                'nx_half': self.nx_half,
                'ny_half': self.ny_half,

                'nx_fourth': self.nx_fourth,
                'ny_fourth': self.ny_fourth,

                'nx_eighth': self.nx_eighth,
                'ny_eighth': self.ny_eighth,

                'nx_sixteenth': self.nx_sixteenth,
                'ny_sixteenth': self.ny_sixteenth,

                'dx': self.dx,
                'dy': self.dy,
            },

            # ---- . ---- ---- ---- ---- . ----
            # radius
            # ---- . ---- ---- ---- ---- . ----
            'radius': {
                'x_v_index': self.x_v_index,
                'y_v_index': self.y_v_index,

                'x_0_index': self.x_0_index,
                'y_0_index': self.y_0_index,

                'x_v': self.x_v,
                'y_v': self.y_v,

                'x_0': self.x_0,
                'y_0': self.y_0,

                'r_v': self.r_v,
                'r_0': self.r_0,
                'r_precision': self.r_precision,
            },

            # ---- . ---- ---- ---- ---- . ----
            # calculated:
            # ---- . ---- ---- ---- ---- . ----
            'calculated': {
                'rho_a_v': self.rho_a_v,
                'rho_b_v': self.rho_b_v,
                'rho_tot_v': self.rho_tot_v,

                'rho_a_0': self.rho_a_0,
                'rho_b_0': self.rho_b_0,
                'rho_tot_0': self.rho_tot_0,

                'delta_norm_0': self.delta_norm_0,

                'mass_star': self.mass_star,

                'k_F': self.k_F,

                'E_F_star': self.E_F_star,

                'T': self.T,

                'T_F': self.T_F,

                'T_c': self.T_c,

                'l_c': self.l_c,

                'a_s': self.a_s,

                'lmb': self.lmb,
            }
        }


    def __str__(
        self
    ):
        return json.dumps(self.__json__(), indent=4)


    def __load_wdata(
        self,
        data_dir: WData,
    ) -> WData:
        return WData.load(data_dir + "/sv-ddcc05p00-T0p05.wtxt")


    def __load_input(
        self,
        data_dir: WData,
    ) -> object:
        handle = open(data_dir + "/input.txt", "r")
        input = handle.read()

        # sclgth
        sclgth = self.__parse_float_from_string(
            r"sclgth *([-+\.\d]+)", input
        )

        # temperature
        temperature = self.__parse_float_from_string(
            r"temperature *([-+\.\d]+)", input
        )

        # return
        return {
            'sclgth': sclgth,
            'temperature': temperature,
        }


    def __parse_float_from_string(
        self,
        pattern,
        input
    ):
        match = re.search(pattern, input)
        if not match:
            raise RuntimeError('could not parse: r"' + str(pattern) + '"')

        return float(match.group(1))


    def __calc_arithmetic_mean_over_circle(
        self,
        data: np.ndarray,
        r: float,
        r_precision: float = 0.01
    ) -> np.float64:
        # Check which points lie on a circle with radius r, with r_precision:
        ijrPoints = self.ijrGrid.get_points_by_radius(r, r_precision)

        # calc data's arithmetic mean value on the circle:
        result = np.float64(0.0)
        for point in ijrPoints:
            i = point.i
            j = point.j
            result += data[i, j]

        result /= len(ijrPoints)

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


#! ---- . ---- ---- ---- ---- . ----
#! gen single subplot:
#! - basic
#! ---- . ---- ---- ---- ---- . ----


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


#! ---- . ---- ---- ---- ---- . ----
#! gen single subplot:
#! - avanced
#! ---- . ---- ---- ---- ---- . ----


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


#! ---- . ---- ---- ---- ---- . ----
#! gen rows of subplots:
#! ---- . ---- ---- ---- ---- . ----


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


#! ---- . ---- ---- ---- ---- . ----
#! gen grids of subplots:
#! ---- . ---- ---- ---- ---- . ----


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


def gen_grid_of_subplots_of_01_series_of_2d_data_vector(
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
    gen_row_of_subplots_of_2d_data_vector(
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


def gen_grid_of_subplots_of_03_series_of_2d_data_vector(
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
    gen_grid_of_subplots_of_01_series_of_2d_data_vector(
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

    gen_grid_of_subplots_of_01_series_of_2d_data_vector(
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

    gen_grid_of_subplots_of_01_series_of_2d_data_vector(
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


#! ---- . ---- ---- ---- ---- . ----
#! gen plots of systen properties:
#! ---- . ---- ---- ---- ---- . ----


def __gen_plot_handle_exception(
    out_file,
    e: Exception
):
    print("failed to plot: " + out_file)
    print(traceback.format_exc())


def gen_plot_01_rho(
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
    try:
        print("drawing: " + out_file)

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

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_02_delta(
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
    try:
        print("drawing: " + out_file)

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

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_03_j(
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

    stride: int,
    scale: float,

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
    try:
        print("drawing: " + out_file)

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

        gen_grid_of_subplots_of_03_series_of_2d_data_vector(
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

            stride = stride,
            scale = scale,

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

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_04_nu(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    nu_norm: np.ndarray,

    title_nu_norm: str,

    label_x: str,
    label_y: str,

    label_nu_norm: str,

    legend_nu_norm: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

            data = nu_norm,

            data_xcsecs_indices = data_xcsecs_indices,
            data_ycsecs_indices = data_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data = title_nu_norm,

            label_x = label_x,
            label_y = label_y,

            label_data = label_nu_norm,

            legend_data = legend_nu_norm,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        # save plot:

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_05_tau(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    tau_tot: np.ndarray,
    tau_a: np.ndarray,
    tau_b: np.ndarray,

    title_tau_tot: str,
    title_tau_a: str,
    title_tau_b: str,

    label_x: str,
    label_y: str,

    label_tau_tot: str,
    label_tau_a: str,
    label_tau_b: str,

    legend_tau_tot: str,
    legend_tau_a: str,
    legend_tau_b: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

            data_01 = tau_tot,
            data_02 = tau_a,
            data_03 = tau_b,

            data_01_xcsecs_indices = data_xcsecs_indices,
            data_01_ycsecs_indices = data_ycsecs_indices,

            data_02_xcsecs_indices = data_xcsecs_indices,
            data_02_ycsecs_indices = data_ycsecs_indices,

            data_03_xcsecs_indices = data_xcsecs_indices,
            data_03_ycsecs_indices = data_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data_01 = title_tau_tot,
            title_data_02 = title_tau_a,
            title_data_03 = title_tau_b,

            label_x = label_x,
            label_y = label_y,

            label_data_01 = label_tau_tot,
            label_data_02 = label_tau_a,
            label_data_03 = label_tau_b,

            legend_data_01 = legend_tau_tot,
            legend_data_02 = legend_tau_a,
            legend_data_03 = legend_tau_b,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_06_V(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    V_tot: np.ndarray,
    V_a: np.ndarray,
    V_b: np.ndarray,

    title_V_tot: str,
    title_V_a: str,
    title_V_b: str,

    label_x: str,
    label_y: str,

    label_V_tot: str,
    label_V_a: str,
    label_V_b: str,

    legend_V_tot: str,
    legend_V_a: str,
    legend_V_b: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

            data_01 = V_tot,
            data_02 = V_a,
            data_03 = V_b,

            data_01_xcsecs_indices = data_xcsecs_indices,
            data_01_ycsecs_indices = data_ycsecs_indices,

            data_02_xcsecs_indices = data_xcsecs_indices,
            data_02_ycsecs_indices = data_ycsecs_indices,

            data_03_xcsecs_indices = data_xcsecs_indices,
            data_03_ycsecs_indices = data_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data_01 = title_V_tot,
            title_data_02 = title_V_a,
            title_data_03 = title_V_b,

            label_x = label_x,
            label_y = label_y,

            label_data_01 = label_V_tot,
            label_data_02 = label_V_a,
            label_data_03 = label_V_b,

            legend_data_01 = legend_V_tot,
            legend_data_02 = legend_V_a,
            legend_data_03 = legend_V_b,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_07_V_ext(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    V_ext_tot: np.ndarray,
    V_ext_a: np.ndarray,
    V_ext_b: np.ndarray,

    title_V_ext_tot: str,
    title_V_ext_a: str,
    title_V_ext_b: str,

    label_x: str,
    label_y: str,

    label_V_ext_tot: str,
    label_V_ext_a: str,
    label_V_ext_b: str,

    legend_V_ext_tot: str,
    legend_V_ext_a: str,
    legend_V_ext_b: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

            data_01 = V_ext_tot,
            data_02 = V_ext_a,
            data_03 = V_ext_b,

            data_01_xcsecs_indices = data_xcsecs_indices,
            data_01_ycsecs_indices = data_ycsecs_indices,

            data_02_xcsecs_indices = data_xcsecs_indices,
            data_02_ycsecs_indices = data_ycsecs_indices,

            data_03_xcsecs_indices = data_xcsecs_indices,
            data_03_ycsecs_indices = data_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data_01 = title_V_ext_tot,
            title_data_02 = title_V_ext_a,
            title_data_03 = title_V_ext_b,

            label_x = label_x,
            label_y = label_y,

            label_data_01 = label_V_ext_tot,
            label_data_02 = label_V_ext_a,
            label_data_03 = label_V_ext_b,

            legend_data_01 = legend_V_ext_tot,
            legend_data_02 = legend_V_ext_a,
            legend_data_03 = legend_V_ext_b,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_08_delta_ext(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    delta_ext_norm: np.ndarray,

    title_delta_ext_norm: str,

    label_x: str,
    label_y: str,

    label_delta_ext_norm: str,

    legend_delta_ext_norm: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

            data = delta_ext_norm,

            data_xcsecs_indices = data_xcsecs_indices,
            data_ycsecs_indices = data_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data = title_delta_ext_norm,

            label_x = label_x,
            label_y = label_y,

            label_data = label_delta_ext_norm,

            legend_data = legend_delta_ext_norm,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        # save plot:

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_09_velocity_ext(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    velocity_ext_tot_norm: np.ndarray,
    velocity_ext_tot_x: np.ndarray,
    velocity_ext_tot_y: np.ndarray,

    velocity_ext_a_norm: np.ndarray,
    velocity_ext_a_x: np.ndarray,
    velocity_ext_a_y: np.ndarray,

    velocity_ext_b_norm: np.ndarray,
    velocity_ext_b_x: np.ndarray,
    velocity_ext_b_y: np.ndarray,

    stride: int,
    scale: float,

    title_velocity_ext_tot: str,
    title_velocity_ext_tot_norm: str,
    title_velocity_ext_tot_x: str,
    title_velocity_ext_tot_y: str,

    title_velocity_ext_a: str,
    title_velocity_ext_a_norm: str,
    title_velocity_ext_a_x: str,
    title_velocity_ext_a_y: str,

    title_velocity_ext_b: str,
    title_velocity_ext_b_norm: str,
    title_velocity_ext_b_x: str,
    title_velocity_ext_b_y: str,

    label_x: str,
    label_y: str,

    label_velocity_ext_tot_norm: str,
    label_velocity_ext_tot_x: str,
    label_velocity_ext_tot_y: str,

    label_velocity_ext_a_norm: str,
    label_velocity_ext_a_x: str,
    label_velocity_ext_a_y: str,

    label_velocity_ext_b_norm: str,
    label_velocity_ext_b_x: str,
    label_velocity_ext_b_y: str,

    legend_velocity_ext_tot_norm: str,
    legend_velocity_ext_tot_x: str,
    legend_velocity_ext_tot_y: str,

    legend_velocity_ext_a_norm: str,
    legend_velocity_ext_a_x: str,
    legend_velocity_ext_a_y: str,

    legend_velocity_ext_b_norm: str,
    legend_velocity_ext_b_x: str,
    legend_velocity_ext_b_y: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

        gen_data_xcsec = lambda data, x: data[x, :]
        gen_data_ycsec = lambda data, y: data[:, y]

        gen_data_xcsec_label = lambda legend, x, i: f"{legend}(x = {x[i]}, y)"
        gen_data_ycsec_label = lambda legend, y, i: f"{legend}(x, y = {y[i]})"

        gen_grid_of_subplots_of_03_series_of_2d_data_vector(
            fig = p.fig,
            ax = p.ax,

            ax_col_offset = 0,

            x = x,
            y = y,

            data_01_norm = velocity_ext_tot_norm,
            data_01_x = velocity_ext_tot_x,
            data_01_y = velocity_ext_tot_y,

            data_02_norm = velocity_ext_a_norm,
            data_02_x = velocity_ext_a_x,
            data_02_y = velocity_ext_a_y,

            data_03_norm = velocity_ext_b_norm,
            data_03_x = velocity_ext_b_x,
            data_03_y = velocity_ext_b_y,

            stride = stride,
            scale = scale,

            data_norm_xcsecs_indices = data_norm_xcsecs_indices,
            data_norm_ycsecs_indices = data_norm_ycsecs_indices,

            data_x_xcsecs_indices = data_x_xcsecs_indices,
            data_x_ycsecs_indices = data_x_ycsecs_indices,

            data_y_xcsecs_indices = data_y_xcsecs_indices,
            data_y_ycsecs_indices = data_y_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data_01 = title_velocity_ext_tot,
            title_data_01_norm = title_velocity_ext_tot_norm,
            title_data_01_x = title_velocity_ext_tot_x,
            title_data_01_y = title_velocity_ext_tot_y,

            title_data_02 = title_velocity_ext_a,
            title_data_02_norm = title_velocity_ext_a_norm,
            title_data_02_x = title_velocity_ext_a_x,
            title_data_02_y = title_velocity_ext_a_y,

            title_data_03 = title_velocity_ext_b,
            title_data_03_norm = title_velocity_ext_b_norm,
            title_data_03_x = title_velocity_ext_b_x,
            title_data_03_y = title_velocity_ext_b_y,

            label_x = label_x,
            label_y = label_y,

            label_data_01_norm = label_velocity_ext_tot_norm,
            label_data_01_x = label_velocity_ext_tot_x,
            label_data_01_y = label_velocity_ext_tot_y,

            label_data_02_norm = label_velocity_ext_a_norm,
            label_data_02_x = label_velocity_ext_a_x,
            label_data_02_y = label_velocity_ext_a_y,

            label_data_03_norm = label_velocity_ext_b_norm,
            label_data_03_x = label_velocity_ext_b_x,
            label_data_03_y = label_velocity_ext_b_y,

            legend_data_01_norm = legend_velocity_ext_tot_norm,
            legend_data_01_x = legend_velocity_ext_tot_x,
            legend_data_01_y = legend_velocity_ext_tot_y,

            legend_data_02_norm = legend_velocity_ext_a_norm,
            legend_data_02_x = legend_velocity_ext_a_x,
            legend_data_02_y = legend_velocity_ext_a_y,

            legend_data_03_norm = legend_velocity_ext_b_norm,
            legend_data_03_x = legend_velocity_ext_b_x,
            legend_data_03_y = legend_velocity_ext_b_y,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        # save plot:

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_10_alpha(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    alpha_tot: np.ndarray,
    alpha_a: np.ndarray,
    alpha_b: np.ndarray,

    title_alpha_tot: str,
    title_alpha_a: str,
    title_alpha_b: str,

    label_x: str,
    label_y: str,

    label_alpha_tot: str,
    label_alpha_a: str,
    label_alpha_b: str,

    legend_alpha_tot: str,
    legend_alpha_a: str,
    legend_alpha_b: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

            data_01 = alpha_tot,
            data_02 = alpha_a,
            data_03 = alpha_b,

            data_01_xcsecs_indices = data_xcsecs_indices,
            data_01_ycsecs_indices = data_ycsecs_indices,

            data_02_xcsecs_indices = data_xcsecs_indices,
            data_02_ycsecs_indices = data_ycsecs_indices,

            data_03_xcsecs_indices = data_xcsecs_indices,
            data_03_ycsecs_indices = data_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data_01 = title_alpha_tot,
            title_data_02 = title_alpha_a,
            title_data_03 = title_alpha_b,

            label_x = label_x,
            label_y = label_y,

            label_data_01 = label_alpha_tot,
            label_data_02 = label_alpha_a,
            label_data_03 = label_alpha_b,

            legend_data_01 = legend_alpha_tot,
            legend_data_02 = legend_alpha_a,
            legend_data_03 = legend_alpha_b,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


def gen_plot_11_A(
    out_file: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    nx: int,
    ny: int,

    x: np.ndarray,
    y: np.ndarray,

    A_tot_norm: np.ndarray,
    A_tot_x: np.ndarray,
    A_tot_y: np.ndarray,

    A_a_norm: np.ndarray,
    A_a_x: np.ndarray,
    A_a_y: np.ndarray,

    A_b_norm: np.ndarray,
    A_b_x: np.ndarray,
    A_b_y: np.ndarray,

    stride: int,
    scale: float,

    title_A_tot: str,
    title_A_tot_norm: str,
    title_A_tot_x: str,
    title_A_tot_y: str,

    title_A_a: str,
    title_A_a_norm: str,
    title_A_a_x: str,
    title_A_a_y: str,

    title_A_b: str,
    title_A_b_norm: str,
    title_A_b_x: str,
    title_A_b_y: str,

    label_x: str,
    label_y: str,

    label_A_tot_norm: str,
    label_A_tot_x: str,
    label_A_tot_y: str,

    label_A_a_norm: str,
    label_A_a_x: str,
    label_A_a_y: str,

    label_A_b_norm: str,
    label_A_b_x: str,
    label_A_b_y: str,

    legend_A_tot_norm: str,
    legend_A_tot_x: str,
    legend_A_tot_y: str,

    legend_A_a_norm: str,
    legend_A_a_x: str,
    legend_A_a_y: str,

    legend_A_b_norm: str,
    legend_A_b_x: str,
    legend_A_b_y: str,

) -> None:
    try:
        print("drawing: " + out_file)

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

        gen_grid_of_subplots_of_03_series_of_2d_data_vector(
            fig = p.fig,
            ax = p.ax,

            ax_col_offset = 0,

            x = x,
            y = y,

            data_01_norm = A_tot_norm,
            data_01_x = A_tot_x,
            data_01_y = A_tot_y,

            data_02_norm = A_a_norm,
            data_02_x = A_a_x,
            data_02_y = A_a_y,

            data_03_norm = A_b_norm,
            data_03_x = A_b_x,
            data_03_y = A_b_y,

            stride = stride,
            scale = scale,

            data_norm_xcsecs_indices = data_norm_xcsecs_indices,
            data_norm_ycsecs_indices = data_norm_ycsecs_indices,

            data_x_xcsecs_indices = data_x_xcsecs_indices,
            data_x_ycsecs_indices = data_x_ycsecs_indices,

            data_y_xcsecs_indices = data_y_xcsecs_indices,
            data_y_ycsecs_indices = data_y_ycsecs_indices,

            gen_data_xcsec_func = gen_data_xcsec,
            gen_data_ycsec_func = gen_data_ycsec,

            title_data_01 = title_A_tot,
            title_data_01_norm = title_A_tot_norm,
            title_data_01_x = title_A_tot_x,
            title_data_01_y = title_A_tot_y,

            title_data_02 = title_A_a,
            title_data_02_norm = title_A_a_norm,
            title_data_02_x = title_A_a_x,
            title_data_02_y = title_A_a_y,

            title_data_03 = title_A_b,
            title_data_03_norm = title_A_b_norm,
            title_data_03_x = title_A_b_x,
            title_data_03_y = title_A_b_y,

            label_x = label_x,
            label_y = label_y,

            label_data_01_norm = label_A_tot_norm,
            label_data_01_x = label_A_tot_x,
            label_data_01_y = label_A_tot_y,

            label_data_02_norm = label_A_a_norm,
            label_data_02_x = label_A_a_x,
            label_data_02_y = label_A_a_y,

            label_data_03_norm = label_A_b_norm,
            label_data_03_x = label_A_b_x,
            label_data_03_y = label_A_b_y,

            legend_data_01_norm = legend_A_tot_norm,
            legend_data_01_x = legend_A_tot_x,
            legend_data_01_y = legend_A_tot_y,

            legend_data_02_norm = legend_A_a_norm,
            legend_data_02_x = legend_A_a_x,
            legend_data_02_y = legend_A_a_y,

            legend_data_03_norm = legend_A_b_norm,
            legend_data_03_x = legend_A_b_x,
            legend_data_03_y = legend_A_b_y,

            gen_data_xcsec_label_func = gen_data_xcsec_label,
            gen_data_ycsec_label_func = gen_data_ycsec_label,
        )

        # save plot:

        p.fig.tight_layout()
        p.fig.savefig(fname = out_file, dpi = dpi)

    except Exception as e:
        __gen_plot_handle_exception(out_file, e)


#! ---- . ---- ---- ---- ---- . ----
#! plot systen properties:
#! ---- . ---- ---- ---- ---- . ----


def plot_01_rho(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_01_rho(
        out_file = out_dir + "/plot 01 - rho - normal density.png",
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

    gen_plot_01_rho(
        out_file = out_dir + "/plot 01 - rho - normal density - normalized.png",
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


def plot_02_delta(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_02_delta(
        out_file = out_dir + "/plot 02 - delta - pairing gap function.png",
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

    gen_plot_02_delta(
        out_file = out_dir + "/plot 02 - delta - pairing gap function - normalized.png",
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


def plot_03_j(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_03_j(
        out_file = out_dir + "/plot 03 - j - current density.png",
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

        stride = 3,
        scale = 0.0005,

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

    # gen_plot_03_j(
    #     out_file = out_dir + "/plot 03 - j - current density - normalized.png",
    #     dpi = dpi,

    #     subplot_w = subplot_w,
    #     subplot_h = subplot_h,

    #     nx = parsed.nx,
    #     ny = parsed.ny,

    #     x = parsed.x_flat,
    #     y = parsed.y_flat,

    #     j_tot_norm = parsed.j_tot_norm / parsed.j_tot_norm_max,
    #     j_tot_x = parsed.j_tot_x / parsed.j_tot_x_max,
    #     j_tot_y = parsed.j_tot_y / parsed.j_tot_y_max,

    #     j_a_norm = parsed.j_a_norm / parsed.j_a_norm_max,
    #     j_a_x = parsed.j_a_x / parsed.j_a_x_max,
    #     j_a_y = parsed.j_a_y / parsed.j_a_y_max,

    #     j_b_norm = parsed.j_b_norm / parsed.j_b_norm_max,
    #     j_b_x = parsed.j_b_x / parsed.j_b_x_max,
    #     j_b_y = parsed.j_b_y / parsed.j_b_y_max,

    #     stride = 3,
    #     scale = 0.1,

    #     title_j_tot = 'j_total / j_total_max',
    #     title_j_tot_norm = '|j_total| / |j_total_max|',
    #     title_j_tot_x = 'j_total_x / j_total_x_max',
    #     title_j_tot_y = 'j_total_y / j_total_y_max',

    #     title_j_a = 'j_a / j_a_max',
    #     title_j_a_norm = '|j_a| / |j_a_max|',
    #     title_j_a_x = 'j_a_x / j_a_x_max',
    #     title_j_a_y = 'j_a_y / j_a_y_max',

    #     title_j_b = 'j_b / j_b_max',
    #     title_j_b_norm = '|j_b| / |j_b_max|',
    #     title_j_b_x = 'j_b_x / j_b_x_max',
    #     title_j_b_y = 'j_b_y / j_b_y_max',

    #     label_x = 'x',
    #     label_y = 'y',

    #     label_j_tot_norm = '|j_total| / |j_total_max|',
    #     label_j_tot_x = 'j_total_x / j_total_x_max',
    #     label_j_tot_y = 'j_total_y / j_total_y_max',

    #     label_j_a_norm = '|j_a| / |j_a_max|',
    #     label_j_a_x = 'j_a_x / j_a_x_max',
    #     label_j_a_y = 'j_a_y / j_a_y_max',

    #     label_j_b_norm = '|j_b| / |j_b_max|',
    #     label_j_b_x = 'j_b_x / j_b_x_max',
    #     label_j_b_y = 'j_b_y / j_b_y_max',

    #     legend_j_tot_norm = '|j_total| / |j_total_max| ',
    #     legend_j_tot_x = 'j_total_x / j_total_x_max ',
    #     legend_j_tot_y = 'j_total_y / j_total_y_max ',

    #     legend_j_a_norm = '|j_a| / |j_a_max| ',
    #     legend_j_a_x = 'j_a_x / j_a_x_max ',
    #     legend_j_a_y = 'j_a_y / j_a_y_max ',

    #     legend_j_b_norm = '|j_b| / |j_b_max| ',
    #     legend_j_b_x = 'j_b_x / j_b_x_max ',
    #     legend_j_b_y = 'j_b_y / j_b_y_max ',
    # )


def plot_04_nu(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_04_nu(
        out_file = out_dir + "/plot 04 - nu.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        nu_norm = parsed.nu_norm,

        title_nu_norm = '|nu|',

        label_x = 'x',
        label_y = 'y',

        label_nu_norm = '|nu|',

        legend_nu_norm = '|nu|',
    )

    gen_plot_04_nu(
        out_file = out_dir + "/plot 04 - nu - normalized.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        nu_norm = parsed.nu_norm / parsed.nu_norm_0,

        title_nu_norm = '|nu| / |nu_0|',

        label_x = 'x',
        label_y = 'y',

        label_nu_norm = '|nu| / |nu_0|',

        legend_nu_norm = '|nu| / |nu_0| ',
    )


def plot_05_tau(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_05_tau(
        out_file = out_dir + "/plot 05 - tau.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        tau_tot = parsed.tau_tot,
        tau_a = parsed.tau_a,
        tau_b = parsed.tau_b,

        title_tau_tot = 'tau_tot',
        title_tau_a = 'tau_a',
        title_tau_b = 'tau_b',

        label_x = 'x',
        label_y = 'y',

        label_tau_tot = 'tau_tot',
        label_tau_a = 'tau_a',
        label_tau_b = 'tau_b',

        legend_tau_tot = 'tau_tot',
        legend_tau_a = 'tau_a',
        legend_tau_b = 'tau_b',
    )

    gen_plot_05_tau(
        out_file = out_dir + "/plot 05 - tau - normalized.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        tau_tot = parsed.tau_tot / parsed.tau_tot_0,
        tau_a = parsed.tau_a / parsed.tau_a_0,
        tau_b = parsed.tau_b / parsed.tau_b_0,

        title_tau_tot = 'tau_total / tau_total_0',
        title_tau_a = 'tau_a / tau_a_0',
        title_tau_b = 'tau_b / tau_b_0',

        label_x = 'x',
        label_y = 'y',

        label_tau_tot = 'tau_total / tau_total_0',
        label_tau_a = 'tau_a / tau_a_0',
        label_tau_b = 'tau_b / tau_b_0',

        legend_tau_tot = 'tau_total / tau_total_0 ',
        legend_tau_a = 'tau_a / tau_a_0 ',
        legend_tau_b = 'tau_b / tau_b_0 ',
    )


def plot_06_V(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_06_V(
        out_file = out_dir + "/plot 06 - V.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        V_tot = parsed.V_tot,
        V_a = parsed.V_a,
        V_b = parsed.V_b,

        title_V_tot = 'V_tot',
        title_V_a = 'V_a',
        title_V_b = 'V_b',

        label_x = 'x',
        label_y = 'y',

        label_V_tot = 'V_tot',
        label_V_a = 'V_a',
        label_V_b = 'V_b',

        legend_V_tot = 'V_tot',
        legend_V_a = 'V_a',
        legend_V_b = 'V_b',
    )

    gen_plot_06_V(
        out_file = out_dir + "/plot 06 - V - normalized.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        V_tot = np.sign(parsed.V_tot) * parsed.V_tot / parsed.V_tot_0,
        V_a = np.sign(parsed.V_a) * parsed.V_a / parsed.V_a_0,
        V_b = np.sign(parsed.V_b) * parsed.V_b / parsed.V_b_0,

        title_V_tot = 'sign(V_total) * V_total / V_total_0 ',
        title_V_a = 'sign(V_a) * V_a / V_a_0 ',
        title_V_b = 'sign(V_b) * V_b / V_b_0 ',

        label_x = 'x',
        label_y = 'y',

        label_V_tot = 'sign(V_total) * V_total / V_total_0 ',
        label_V_a = 'sign(V_a) * V_a / V_a_0 ',
        label_V_b = 'sign(V_b) * V_b / V_b_0 ',

        legend_V_tot = 'sign(V_total) * V_total / V_total_0 ',
        legend_V_a = 'sign(V_a) * V_a / V_a_0 ',
        legend_V_b = 'sign(V_b) * V_b / V_b_0 ',
    )


def plot_07_V_ext(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_07_V_ext(
        out_file = out_dir + "/plot 07 - V_ext.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        V_ext_tot = parsed.V_ext_tot,
        V_ext_a = parsed.V_ext_a,
        V_ext_b = parsed.V_ext_b,

        title_V_ext_tot = 'V_ext_tot',
        title_V_ext_a = 'V_ext_a',
        title_V_ext_b = 'V_ext_b',

        label_x = 'x',
        label_y = 'y',

        label_V_ext_tot = 'V_ext_tot',
        label_V_ext_a = 'V_ext_a',
        label_V_ext_b = 'V_ext_b',

        legend_V_ext_tot = 'V_ext_tot',
        legend_V_ext_a = 'V_ext_a',
        legend_V_ext_b = 'V_ext_b',
    )

    # gen_plot_07_V_ext(
    #     out_file = out_dir + "/plot 07 - V_ext - normalized.png",
    #     dpi = dpi,

    #     subplot_w = subplot_w,
    #     subplot_h = subplot_h,

    #     nx = parsed.nx,
    #     ny = parsed.ny,

    #     x = parsed.x_flat,
    #     y = parsed.y_flat,

    #     V_ext_tot = parsed.V_ext_tot / parsed.V_ext_tot_0,
    #     V_ext_a = parsed.V_ext_a / parsed.V_ext_a_0,
    #     V_ext_b = parsed.V_ext_b / parsed.V_ext_b_0,

    #     title_V_ext_tot = 'V_ext_total / V_ext_total_0',
    #     title_V_ext_a = 'V_ext_a / V_ext_a_0',
    #     title_V_ext_b = 'V_ext_b / V_ext_b_0',

    #     label_x = 'x',
    #     label_y = 'y',

    #     label_V_ext_tot = 'V_ext_total / V_ext_total_0',
    #     label_V_ext_a = 'V_ext_a / V_ext_a_0',
    #     label_V_ext_b = 'V_ext_b / V_ext_b_0',

    #     legend_V_ext_tot = 'V_ext_total / V_ext_total_0 ',
    #     legend_V_ext_a = 'V_ext_a / V_ext_a_0 ',
    #     legend_V_ext_b = 'V_ext_b / V_ext_b_0 ',
    # )


def plot_08_delta_ext(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_08_delta_ext(
        out_file = out_dir + "/plot 08 - delta_ext.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        delta_ext_norm = parsed.delta_ext_norm,

        title_delta_ext_norm = '|delta_ext|',

        label_x = 'x',
        label_y = 'y',

        label_delta_ext_norm = '|delta_ext|',

        legend_delta_ext_norm = '|delta_ext|',
    )

    # gen_plot_08_delta_ext(
    #     out_file = out_dir + "/plot 08 - delta_ext - normalized.png",
    #     dpi = dpi,

    #     subplot_w = subplot_w,
    #     subplot_h = subplot_h,

    #     nx = parsed.nx,
    #     ny = parsed.ny,

    #     x = parsed.x_flat,
    #     y = parsed.y_flat,

    #     delta_ext_norm = parsed.delta_ext_norm / parsed.delta_ext_norm_0,

    #     title_delta_ext_norm = '|delta_ext| / |delta_ext_0|',

    #     label_x = 'x',
    #     label_y = 'y',

    #     label_delta_ext_norm = '|delta_ext| / |delta_ext_0|',

    #     legend_delta_ext_norm = '|delta_ext| / |delta_ext_0| ',
    # )


def plot_09_velocity_ext(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_09_velocity_ext(
        out_file = out_dir + "/plot 09 - velocity_ext.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        velocity_ext_tot_norm = parsed.velocity_ext_tot_norm,
        velocity_ext_tot_x = parsed.velocity_ext_tot_x,
        velocity_ext_tot_y = parsed.velocity_ext_tot_y,

        velocity_ext_a_norm = parsed.velocity_ext_a_norm,
        velocity_ext_a_x = parsed.velocity_ext_a_x,
        velocity_ext_a_y = parsed.velocity_ext_a_y,

        velocity_ext_b_norm = parsed.velocity_ext_b_norm,
        velocity_ext_b_x = parsed.velocity_ext_b_x,
        velocity_ext_b_y = parsed.velocity_ext_b_y,

        stride = 3,
        scale = 0.0005,

        title_velocity_ext_tot = 'velocity_ext_total',
        title_velocity_ext_tot_norm = '|velocity_ext_total|',
        title_velocity_ext_tot_x = 'velocity_ext_total_x',
        title_velocity_ext_tot_y = 'velocity_ext_total_y',

        title_velocity_ext_a = 'velocity_ext_a',
        title_velocity_ext_a_norm = '|velocity_ext_a|',
        title_velocity_ext_a_x = 'velocity_ext_a_x',
        title_velocity_ext_a_y = 'velocity_ext_a_y',

        title_velocity_ext_b = 'velocity_ext_b',
        title_velocity_ext_b_norm = '|velocity_ext_b|',
        title_velocity_ext_b_x = 'velocity_ext_b_x',
        title_velocity_ext_b_y = 'velocity_ext_b_y',

        label_x = 'x',
        label_y = 'y',

        label_velocity_ext_tot_norm = '|velocity_ext_total|',
        label_velocity_ext_tot_x = 'velocity_ext_total_x',
        label_velocity_ext_tot_y = 'velocity_ext_total_y',

        label_velocity_ext_a_norm = '|velocity_ext_a|',
        label_velocity_ext_a_x = 'velocity_ext_a_x',
        label_velocity_ext_a_y = 'velocity_ext_a_y',

        label_velocity_ext_b_norm = '|velocity_ext_b|',
        label_velocity_ext_b_x = 'velocity_ext_b_x',
        label_velocity_ext_b_y = 'velocity_ext_b_y',

        legend_velocity_ext_tot_norm = '|velocity_ext_total|',
        legend_velocity_ext_tot_x = 'velocity_ext_total_x',
        legend_velocity_ext_tot_y = 'velocity_ext_total_y',

        legend_velocity_ext_a_norm = '|velocity_ext_a|',
        legend_velocity_ext_a_x = 'velocity_ext_a_x',
        legend_velocity_ext_a_y = 'velocity_ext_a_y',

        legend_velocity_ext_b_norm = '|velocity_ext_b|',
        legend_velocity_ext_b_x = 'velocity_ext_b_x',
        legend_velocity_ext_b_y = 'velocity_ext_b_y',
    )

    # gen_plot_09_velocity_ext(
    #     out_file = out_dir + "/plot 09 - velocity_ext - normalized.png",
    #     dpi = dpi,

    #     subplot_w = subplot_w,
    #     subplot_h = subplot_h,

    #     nx = parsed.nx,
    #     ny = parsed.ny,

    #     x = parsed.x_flat,
    #     y = parsed.y_flat,

    #     velocity_ext_tot_norm = parsed.velocity_ext_tot_norm / parsed.velocity_ext_tot_norm_max,
    #     velocity_ext_tot_x = parsed.velocity_ext_tot_x / parsed.velocity_ext_tot_x_max,
    #     velocity_ext_tot_y = parsed.velocity_ext_tot_y / parsed.velocity_ext_tot_y_max,

    #     velocity_ext_a_norm = parsed.velocity_ext_a_norm / parsed.velocity_ext_a_norm_max,
    #     velocity_ext_a_x = parsed.velocity_ext_a_x / parsed.velocity_ext_a_x_max,
    #     velocity_ext_a_y = parsed.velocity_ext_a_y / parsed.velocity_ext_a_y_max,

    #     velocity_ext_b_norm = parsed.velocity_ext_b_norm / parsed.velocity_ext_b_norm_max,
    #     velocity_ext_b_x = parsed.velocity_ext_b_x / parsed.velocity_ext_b_x_max,
    #     velocity_ext_b_y = parsed.velocity_ext_b_y / parsed.velocity_ext_b_y_max,

    #     stride = 3,
    #     scale = 0.1,

    #     title_velocity_ext_tot = 'velocity_ext_total / velocity_ext_total_max',
    #     title_velocity_ext_tot_norm = '|velocity_ext_total| / |velocity_ext_total_max|',
    #     title_velocity_ext_tot_x = 'velocity_ext_total_x / velocity_ext_total_x_max',
    #     title_velocity_ext_tot_y = 'velocity_ext_total_y / velocity_ext_total_y_max',

    #     title_velocity_ext_a = 'velocity_ext_a / velocity_ext_a_max',
    #     title_velocity_ext_a_norm = '|velocity_ext_a| / |velocity_ext_a_max|',
    #     title_velocity_ext_a_x = 'velocity_ext_a_x / velocity_ext_a_x_max',
    #     title_velocity_ext_a_y = 'velocity_ext_a_y / velocity_ext_a_y_max',

    #     title_velocity_ext_b = 'velocity_ext_b / velocity_ext_b_max',
    #     title_velocity_ext_b_norm = '|velocity_ext_b| / |velocity_ext_b_max|',
    #     title_velocity_ext_b_x = 'velocity_ext_b_x / velocity_ext_b_x_max',
    #     title_velocity_ext_b_y = 'velocity_ext_b_y / velocity_ext_b_y_max',

    #     label_x = 'x',
    #     label_y = 'y',

    #     label_velocity_ext_tot_norm = '|velocity_ext_total| / |velocity_ext_total_max|',
    #     label_velocity_ext_tot_x = 'velocity_ext_total_x / velocity_ext_total_x_max',
    #     label_velocity_ext_tot_y = 'velocity_ext_total_y / velocity_ext_total_y_max',

    #     label_velocity_ext_a_norm = '|velocity_ext_a| / |velocity_ext_a_max|',
    #     label_velocity_ext_a_x = 'velocity_ext_a_x / velocity_ext_a_x_max',
    #     label_velocity_ext_a_y = 'velocity_ext_a_y / velocity_ext_a_y_max',

    #     label_velocity_ext_b_norm = '|velocity_ext_b| / |velocity_ext_b_max|',
    #     label_velocity_ext_b_x = 'velocity_ext_b_x / velocity_ext_b_x_max',
    #     label_velocity_ext_b_y = 'velocity_ext_b_y / velocity_ext_b_y_max',

    #     legend_velocity_ext_tot_norm = '|velocity_ext_total| / |velocity_ext_total_max| ',
    #     legend_velocity_ext_tot_x = 'velocity_ext_total_x / velocity_ext_total_x_max ',
    #     legend_velocity_ext_tot_y = 'velocity_ext_total_y / velocity_ext_total_y_max ',

    #     legend_velocity_ext_a_norm = '|velocity_ext_a| / |velocity_ext_a_max| ',
    #     legend_velocity_ext_a_x = 'velocity_ext_a_x / velocity_ext_a_x_max ',
    #     legend_velocity_ext_a_y = 'velocity_ext_a_y / velocity_ext_a_y_max ',

    #     legend_velocity_ext_b_norm = '|velocity_ext_b| / |velocity_ext_b_max| ',
    #     legend_velocity_ext_b_x = 'velocity_ext_b_x / velocity_ext_b_x_max ',
    #     legend_velocity_ext_b_y = 'velocity_ext_b_y / velocity_ext_b_y_max ',
    # )


def plot_10_alpha(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_10_alpha(
        out_file = out_dir + "/plot 10 - alpha.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        alpha_tot = parsed.alpha_tot,
        alpha_a = parsed.alpha_a,
        alpha_b = parsed.alpha_b,

        title_alpha_tot = 'alpha_tot',
        title_alpha_a = 'alpha_a',
        title_alpha_b = 'alpha_b',

        label_x = 'x',
        label_y = 'y',

        label_alpha_tot = 'alpha_tot',
        label_alpha_a = 'alpha_a',
        label_alpha_b = 'alpha_b',

        legend_alpha_tot = 'alpha_tot',
        legend_alpha_a = 'alpha_a',
        legend_alpha_b = 'alpha_b',
    )

    gen_plot_10_alpha(
        out_file = out_dir + "/plot 10 - alpha - normalized.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        alpha_tot = parsed.alpha_tot / parsed.alpha_tot_0,
        alpha_a = parsed.alpha_a / parsed.alpha_a_0,
        alpha_b = parsed.alpha_b / parsed.alpha_b_0,

        title_alpha_tot = 'alpha_total / alpha_total_0',
        title_alpha_a = 'alpha_a / alpha_a_0',
        title_alpha_b = 'alpha_b / alpha_b_0',

        label_x = 'x',
        label_y = 'y',

        label_alpha_tot = 'alpha_total / alpha_total_0',
        label_alpha_a = 'alpha_a / alpha_a_0',
        label_alpha_b = 'alpha_b / alpha_b_0',

        legend_alpha_tot = 'alpha_total / alpha_total_0 ',
        legend_alpha_a = 'alpha_a / alpha_a_0 ',
        legend_alpha_b = 'alpha_b / alpha_b_0 ',
    )


def plot_11_A(
    out_dir: str,
    dpi: int,

    subplot_w: int,
    subplot_h: int,

    parsed: ParsedData
) -> None:
    gen_plot_11_A(
        out_file = out_dir + "/plot 11 - A.png",
        dpi = dpi,

        subplot_w = subplot_w,
        subplot_h = subplot_h,

        nx = parsed.nx,
        ny = parsed.ny,

        x = parsed.x_flat,
        y = parsed.y_flat,

        A_tot_norm = parsed.A_tot_norm,
        A_tot_x = parsed.A_tot_x,
        A_tot_y = parsed.A_tot_y,

        A_a_norm = parsed.A_a_norm,
        A_a_x = parsed.A_a_x,
        A_a_y = parsed.A_a_y,

        A_b_norm = parsed.A_b_norm,
        A_b_x = parsed.A_b_x,
        A_b_y = parsed.A_b_y,

        stride = 3,
        scale = 0.005,

        title_A_tot = 'A_total',
        title_A_tot_norm = '|A_total|',
        title_A_tot_x = 'A_total_x',
        title_A_tot_y = 'A_total_y',

        title_A_a = 'A_a',
        title_A_a_norm = '|A_a|',
        title_A_a_x = 'A_a_x',
        title_A_a_y = 'A_a_y',

        title_A_b = 'A_b',
        title_A_b_norm = '|A_b|',
        title_A_b_x = 'A_b_x',
        title_A_b_y = 'A_b_y',

        label_x = 'x',
        label_y = 'y',

        label_A_tot_norm = '|A_total|',
        label_A_tot_x = 'A_total_x',
        label_A_tot_y = 'A_total_y',

        label_A_a_norm = '|A_a|',
        label_A_a_x = 'A_a_x',
        label_A_a_y = 'A_a_y',

        label_A_b_norm = '|A_b|',
        label_A_b_x = 'A_b_x',
        label_A_b_y = 'A_b_y',

        legend_A_tot_norm = '|A_total|',
        legend_A_tot_x = 'A_total_x',
        legend_A_tot_y = 'A_total_y',

        legend_A_a_norm = '|A_a|',
        legend_A_a_x = 'A_a_x',
        legend_A_a_y = 'A_a_y',

        legend_A_b_norm = '|A_b|',
        legend_A_b_x = 'A_b_x',
        legend_A_b_y = 'A_b_y',
    )

    # gen_plot_11_A(
    #     out_file = out_dir + "/plot 11 - A - normalized.png",
    #     dpi = dpi,

    #     subplot_w = subplot_w,
    #     subplot_h = subplot_h,

    #     nx = parsed.nx,
    #     ny = parsed.ny,

    #     x = parsed.x_flat,
    #     y = parsed.y_flat,

    #     A_tot_norm = parsed.A_tot_norm / parsed.A_tot_norm_max,
    #     A_tot_x = parsed.A_tot_x / parsed.A_tot_x_max,
    #     A_tot_y = parsed.A_tot_y / parsed.A_tot_y_max,

    #     A_a_norm = parsed.A_a_norm / parsed.A_a_norm_max,
    #     A_a_x = parsed.A_a_x / parsed.A_a_x_max,
    #     A_a_y = parsed.A_a_y / parsed.A_a_y_max,

    #     A_b_norm = parsed.A_b_norm / parsed.A_b_norm_max,
    #     A_b_x = parsed.A_b_x / parsed.A_b_x_max,
    #     A_b_y = parsed.A_b_y / parsed.A_b_y_max,

    #     stride = 3,
    #     scale = 0.1,

    #     title_A_tot = 'A_total / A_total_max',
    #     title_A_tot_norm = '|A_total| / |A_total_max|',
    #     title_A_tot_x = 'A_total_x / A_total_x_max',
    #     title_A_tot_y = 'A_total_y / A_total_y_max',

    #     title_A_a = 'A_a / A_a_max',
    #     title_A_a_norm = '|A_a| / |A_a_max|',
    #     title_A_a_x = 'A_a_x / A_a_x_max',
    #     title_A_a_y = 'A_a_y / A_a_y_max',

    #     title_A_b = 'A_b / A_b_max',
    #     title_A_b_norm = '|A_b| / |A_b_max|',
    #     title_A_b_x = 'A_b_x / A_b_x_max',
    #     title_A_b_y = 'A_b_y / A_b_y_max',

    #     label_x = 'x',
    #     label_y = 'y',

    #     label_A_tot_norm = '|A_total| / |A_total_max|',
    #     label_A_tot_x = 'A_total_x / A_total_x_max',
    #     label_A_tot_y = 'A_total_y / A_total_y_max',

    #     label_A_a_norm = '|A_a| / |A_a_max|',
    #     label_A_a_x = 'A_a_x / A_a_x_max',
    #     label_A_a_y = 'A_a_y / A_a_y_max',

    #     label_A_b_norm = '|A_b| / |A_b_max|',
    #     label_A_b_x = 'A_b_x / A_b_x_max',
    #     label_A_b_y = 'A_b_y / A_b_y_max',

    #     legend_A_tot_norm = '|A_total| / |A_total_max| ',
    #     legend_A_tot_x = 'A_total_x / A_total_x_max ',
    #     legend_A_tot_y = 'A_total_y / A_total_y_max ',

    #     legend_A_a_norm = '|A_a| / |A_a_max| ',
    #     legend_A_a_x = 'A_a_x / A_a_x_max ',
    #     legend_A_a_y = 'A_a_y / A_a_y_max ',

    #     legend_A_b_norm = '|A_b| / |A_b_max| ',
    #     legend_A_b_x = 'A_b_x / A_b_x_max ',
    #     legend_A_b_y = 'A_b_y / A_b_y_max ',
    # )


pass
#! ---- . ---- ---- ---- ---- . ----
#! plot systen properties:
#! ---- . ---- ---- ---- ---- . ----


def plot(
    parsed: ParsedData,
    outpud_dir: str
):

    subplot_w = 7
    subplot_h = 7

    dpi = 300

    plot_01_rho(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_02_delta(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_03_j(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_04_nu(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_05_tau(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_06_V(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_07_V_ext(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_07_V_ext(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_08_delta_ext(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_09_velocity_ext(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_10_alpha(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )

    plot_11_A(
        out_dir = outpud_dir,
        dpi = dpi,
        subplot_w = subplot_w,
        subplot_h = subplot_h,
        parsed = parsed
    )


#! ---- . ---- ---- ---- ---- . ----
#! start:
#! ---- . ---- ---- ---- ---- . ----

def main() -> int:
    results_dir = '/workspace/results'
    input_dir = results_dir + '/data'
    output_dir = results_dir + '/analysis'
    for simulation in os.listdir(input_dir):
        # skip inalid dirs:
        if not re.search(
            r"st_vortex_01_data_\d+_input_80_80_\d+\.\d+",
            simulation
        ):
            continue

        # gen in/out dir paths:
        simulation_input_dir = input_dir + '/' + simulation
        simulation_output_dir = output_dir + '/' + simulation

        # create out dir:
        try:
            os.mkdir(simulation_output_dir)
        except FileExistsError:
            pass # do nothing

        # check if input dir exists:
        if not os.path.exists(simulation_input_dir):
            raise RuntimeError('input dir does not exist')

        # check if input dir exists:
        if not os.path.exists(simulation_output_dir):
            raise RuntimeError('output dir does not exist')

        # parse data
        parsed = ParsedData(simulation_input_dir)
        print(parsed)

        # # plot
        # plot(parsed, simulation_output_dir)

    return 0

if __name__ == '__main__':
    sys.exit(main())
