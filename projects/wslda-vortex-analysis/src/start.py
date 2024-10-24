#!/usr/bin/python3

from typing import Any, Union, Tuple, List, Set, Dict
import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from wdata.io import WData, Var


def plot_pcolormesh(
    data,
    title,
    xlabel,
    ylabel,
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
    x_crossection_indices: List[int],
    y_crossection_indices: List[int]
):
    # helpers:
    iteration = -1
    x_axis = 0
    y_axis = 1
    x_axis = 0
    y_axis = 1

    # grid:
    x = data.xyz[x_axis]
    y = data.xyz[y_axis]
    x_flat = x.flatten()
    y_flat = y.flatten()
    Nx = data.Nxyz[x_axis]
    Ny = data.Nxyz[y_axis]
    pprint.pp(x_flat)
    pprint.pp(y_flat)
    pprint.pp(Nx)
    pprint.pp(Ny)

    # test = np.array([
    #     [ 1, 2, 3 ],
    #     [ 4, 5, 6 ],
    #     [ 7, 8, 9 ]
    # ])
    # pprint.pp(test[0, :])
    # pprint.pp(test[:, 0])

    # data:
    j_a_x = data.j_a[iteration][x_axis]
    j_a_y = data.j_a[iteration][y_axis]
    j_b_x = data.j_b[iteration][x_axis]
    j_b_y = data.j_b[iteration][y_axis]

    # create plot with sub plots:
    subplot_h = 10
    subplot_w = 10
    fig_nrows = 1
    fig_ncols = 3
    fig_row_ratios = [1, 1, 1]
    fig_col_ratios = [1]
    fig_w = np.sum([subplot_w * ratio / np.max(fig_row_ratios) for ratio in fig_row_ratios])
    fig_h = np.sum([subplot_h * ratio / np.max(fig_col_ratios) for ratio in fig_col_ratios])
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        nrows=fig_nrows,
        ncols=fig_ncols,
        width_ratios=fig_row_ratios,
        height_ratios=fig_col_ratios
    )
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # First subplot:
    plot0 = ax0.pcolormesh(x_flat, y_flat, j_a_x, cmap='plasma', shading='auto')
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_title('j_a_x')
    ax0.set_xlabel('y')
    ax0.set_ylabel('j_a_x')
    fig.colorbar(
        plot0, ax=ax0, location='left', orientation='vertical', fraction=0.035, pad=-0.16
    )

    fig.tight_layout()
    fig.savefig("plot.png", dpi=300)

    # # crossections:
    # for i in [35, 38, 40, 42, 45]:
    #     j_a_x = data.j_a[iteration][x_axis][i, :]
    #     plt.plot(x, j_a_x, label=f"j_a_x(x = {x_flat[i]:.2f}, y)")
    # plt.legend(loc='upper right')
    # plt.xlabel('y')
    # plt.ylabel('j_a_x')
    # plt.title('j_a_x')
    # plt.savefig(f"plot.png")
    # plt.cla()

    # rho_a = data.rho_a[iteration][int(Nx/2),:]
    # rho_b = data.rho_b[iteration][int(Nx/2),:]
    # j_b_x = data.j_b[iteration][x_axis][int(Nx/2)]
    # j_b_y = data.j_b[iteration][y_axis][int(Ny/2)]

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
    x_axis = 0
    y_axis = 1

    x_crossection_indices = [20, 30, 40, 50, 60],
    y_crossection_indices = [20, 30, 40, 50, 60],

    # plot:
    plot_current_density(
        data = data,
        x_crossection_indices = x_crossection_indices,
        y_crossection_indices = y_crossection_indices,
    )

    return 0

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.
    sys.exit(main())
