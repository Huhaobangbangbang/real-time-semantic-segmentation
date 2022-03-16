"""
 -*- coding: utf-8 -*-
 author： Hao Hu
 @date   2022/3/12 9:04 PM
"""

import os
import logging

import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt

import morphsnakes as ms


def visual_callback_2d(background, fig=None):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


def visual_callback_3d(fig=None, plot_each=1):

    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.pause(0.001)

    counter = [-1]

    def callback(levelset):

        counter[0] += 1
        if (counter[0] % plot_each) != 0:
            return

        if ax.collections:
            del ax.collections[0]

        coords, triangles = mcubes.marching_cubes(levelset, 0.5)
        ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                        triangles=triangles)
        plt.pause(0.1)

    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def example_flaw_edge(flaw_path):

    # Load the image.
    imgcolor = imread(flaw_path)/255.0
    img = rgb2gray(imgcolor)

    # MorphACWE does not need g(I)

    # Initialization of the level-set.
    init_ls = ms.circle_level_set(img.shape, (80, 170), 25)

    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # Morphological Chan-Vese (or ACWE)
    ms.morphological_chan_vese(img, iterations=200,
                               init_level_set=init_ls,
                               smoothing=3, lambda1=1, lambda2=1,
                               iter_callback=callback)

if __name__ == '__main__':
    flaw_path = '/Users/huhao/Documents/GitHub/real-time-semantic-segmentation/PCB/pcb_small/20211022_10583-0-01-7_fake_B.png' # (245, 256)
    example_flaw_edge(flaw_path)
