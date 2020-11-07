import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import math
from helper import convolution

mask7 = np.array([[1,1,2,2,2,1,1], [1,2,2,4,2,2,1], [2,2,4,8,4,2,2], [2,4,8,16,8,4,2], [2,2,4,8,4,2,2], [1,2,2,4,2,2,1], [1,1,2,2,2,1,1]])
mask15 = np.array([[2,2,3,4,5,5,6,6,6,5,5,4,3,2,2], [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2], [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3], [4,5,7,9,10,12,13,13,13,12,10,9,7,5,4], [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5], [5,7,10,12,14,16,17,18,17, 16,14,12,10,7,5], [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6], [6,8,11,13,16,18,19,20,19,18, 16,13, 11, 8, 6], [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6], [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5], [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5], [4,5,7,9,10,12,13,13,13,12,10, 9, 7,5, 4], [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3], [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2], [2,2,3,4,5,5,6,6,6,5,5,4,3,2,2]])
# for r in mask7:
#     for c in r:
#         print(c,end = " ")
#     print()

# print()
# print()
# print()

# for r in mask15:
#     for c in r:
#         print(c,end = " ")
#     print()

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


# def gaussian_kernel(size, sigma=1, verbose=False):
#     kernel_1D = np.linspace(-(size // 2), size // 2, size)
#     for i in range(size):
#         kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
#     kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

#     kernel_2D *= 1.0 / kernel_2D.max()

#     if verbose:
#         plt.imshow(kernel_2D, interpolation='none', cmap='gray')
#         plt.title("Kernel ( {}X{} )".format(size, size))
#         plt.show()

#     return kernel_2D


def gaussian_blur(image, kernel, verbose=False):
    return convolution(image, kernel, average=True, verbose=verbose)


if __name__ == '__main__':
    image = cv2.imread("lenna.jpg")

    gaussian_blur(image, mask15, verbose=True)