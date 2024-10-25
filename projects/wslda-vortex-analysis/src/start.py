#!/usr/bin/python3

from typing import Any, Union, Tuple, List, Set, Dict
import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from enum import Enum
from scipy.interpolate import interp1d
from wdata.io import WData, Var


class Axis(int, Enum):
    X = 0
    Y = 1
    Z = 2

class Component(int, Enum):
    X = 0
    Y = 1
    Z = 2

class Grid:
    def __init__(self, data: WData):
        self.Nx = data.Nxyz[Axis.X]
        self.Ny = data.Nxyz[Axis.Y]
        self.dx = data.dxyz[Axis.X]
        self.dy = data.dxyz[Axis.Y]
        self.x = data.xyz[Axis.X]
        self.y = data.xyz[Axis.Y]
        self.x_flat = self.x.flatten()
        self.y_flat = self.y.flatten()

def plot_pcolormesh(
    data,
    title,
    label_x,
    label_y,
):
    # First subplot:
    plt.pcolormesh(data, cmap='plasma', shading='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)
    plt.savefig("plot.png", dpi=300)
    plt.cla()

def plot_current_density(
    data: WData,
    iteration: int,
    x_crossection_indices: List[int],
    y_crossection_indices: List[int]
):

    # data:
    # - grid:
    Nx = data.Nxyz[Axis.X]
    Ny = data.Nxyz[Axis.Y]
    dx = data.dxyz[Axis.X]
    dy = data.dxyz[Axis.Y]
    x = data.xyz[Axis.X]
    y = data.xyz[Axis.Y]
    x_flat = x.flatten()
    y_flat = y.flatten()

    # data:
    # - current density
    j_a_x = data.j_a[iteration][Component.X]
    j_a_y = data.j_a[iteration][Component.Y]
    j_b_x = data.j_b[iteration][Component.X]
    j_b_y = data.j_b[iteration][Component.Y]

    # plot's width and height:
    subplot_w = 7
    subplot_h = 7
    plot_ncols = 3
    plot_nrows = 1
    plot_cols_w_ratios = [1, 1, 1]
    plot_rows_h_ratios = [1]
    plot_w = np.sum([subplot_w * ratio for ratio in (plot_cols_w_ratios / np.max(plot_cols_w_ratios))])
    plot_h = np.sum([subplot_h * ratio for ratio in (plot_rows_h_ratios / np.max(plot_rows_h_ratios))])

    # create plot and subplots:
    figure = plt.figure(figsize=(plot_w, plot_h))
    grid = gridspec.GridSpec(
        nrows=plot_nrows,
        ncols=plot_ncols,
        width_ratios=plot_cols_w_ratios,
        height_ratios=plot_rows_h_ratios
    )
    axis_0 = figure.add_subplot(grid[0])
    axis_1 = figure.add_subplot(grid[1])
    axis_2 = figure.add_subplot(grid[2])

    # first subplot:
    pcolormesh_0 = axis_0.pcolormesh(x_flat, y_flat, j_a_x, cmap='plasma', shading='auto')
    axis_0.set_aspect('equal', adjustable='box')
    axis_0.set_title('j_a_x')
    axis_0.set_xlabel('x')
    axis_0.set_ylabel('y')
    figure.colorbar(
        pcolormesh_0,
        ax=axis_0,

        # location='left',
        # orientation='vertical',
        # fraction=0.035,
        # pad=-0.16

        location='right',
        orientation='vertical',
        fraction=0.035,
        pad=0.01,
    )

    figure.tight_layout()
    figure.savefig(
        fname="plot.png",
        dpi=300
    )

    # # crossections:
    # for i in [35, 38, 40, 42, 45]:
    #     j_a_x = data.j_a[iteration][Axis.X][i, :]
    #     plt.plot(x, j_a_x, label=f"j_a_x(x = {x_flat[i]:.2f}, y)")
    # plt.legend(loc='upper right')
    # plt.xlabel('y')
    # plt.ylabel('j_a_x')
    # plt.title('j_a_x')
    # plt.savefig(f"plot.png")
    # plt.cla()

    # rho_a = data.rho_a[iteration][int(Nx/2),:]
    # rho_b = data.rho_b[iteration][int(Nx/2),:]
    # j_b_x = data.j_b[iteration][Axis.X][int(Nx/2)]
    # j_b_y = data.j_b[iteration][Axis.Y][int(Ny/2)]

def test_02(data: WData):
    j_a = data.j_a[-1] # last frame
    rho_a = data.rho_a[-1] # last frame

    x = data.xyz[0]
    sec=int(data.Nxyz[0]/2)
    j_a = j_a[0,sec,:]      # extract x component of j_a along y axis for x=Nx/2 (center of box)
    rho_a = rho_a[sec,:]    # extract rho_a along y axis for x=Nx/2 (center of box)
    vs = np.abs(j_a/rho_a)  # computer velocity (absolute value)

    # # interpolate data
    # vs_int = interp1d(x[:,0], vs, kind='quadratic')
    # newx=np.linspace(-20,20,200)

    # # flow for ideal vortex
    # r = np.linspace(3,40,100)    # take range [3-40]
    # vs_ideal = 0.5 / r           # vs=1/2r (m=hbar=1, and factor 2 accounts for Cooper pairs)

    # # plot
    # fig, ax = plt.subplots()
    # ax.plot(x, vs, 'bo', markersize=3, label='data')
    # ax.plot(newx, vs_int(newx), 'b', linestyle="-")
    # ax.plot(r, vs_ideal, color="k", linestyle="--", label=r'$\sim 1/r$')
    # ax.plot(r*(-1.0), vs_ideal, color="k", linestyle="--")

    # ax.set(xlabel='y', ylabel=r'$v(0,y)$')
    # ax.set_xlim([-20,20])
    # ax.grid()
    # plt.legend(loc='upper right')

    # fig.savefig("velocity.png")

def main() -> int:
    """
    entry point
    """

    inDir = "/workspace/results/data"
    outDir = "/workspace/results/analysis"
    inFile = inDir + "/st-vortex-recreation-01/sv-ddcc05p00-T0p05.wtxt"
    data = WData.load(inFile)

    # pprint.pp(vars(data))
    # pprint.pp(data.aliases)
    # pprint.pp(data.constants)

    # helpers:
    iteration = -1
    x_crossection_indices = [20, 30, 40, 50, 60],
    y_crossection_indices = [20, 30, 40, 50, 60],

    # plot:
    plot_current_density(
        data = data,
        iteration = iteration,
        x_crossection_indices = x_crossection_indices,
        y_crossection_indices = y_crossection_indices,
    )

    return 0

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    sys.exit(main())
