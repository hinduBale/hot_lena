import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output


# def convolution(oldimage, kernel):
#     #image = Image.fromarray(image, 'RGB')
#     image_h = oldimage.shape[0]
#     image_w = oldimage.shape[1]
    
    
#     kernel_h = kernel.shape[0]
#     kernel_w = kernel.shape[1]
    
#     if(len(oldimage.shape) == 3):
#         image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2),(0,0)), mode='constant',constant_values=0).astype(np.float32)    
#     elif(len(oldimage.shape) == 2):
#         image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2),(kernel_w // 2, kernel_w // 2)), mode='constant', constant_values=0).astype(np.float32)
    
    
#     h = kernel_h // 2
#     w = kernel_w // 2
    
#     image_conv = np.zeros(image_pad.shape)
    
#     for i in range(h, image_pad.shape[0]-h):
#         for j in range(w, image_pad.shape[1]-w):
#             #sum = 0
#             x = image_pad[i-h:i-h+kernel_h, j-w:j-w+kernel_w]
#             x = x.flatten()*kernel.flatten()
#             image_conv[i][j] = x.sum()
#     h_end = -h
#     w_end = -w
    
#     if(h == 0):
#         return image_conv[h:,w:w_end]
#     if(w == 0):
#         return image_conv[h:h_end,w:]
    
#     return image_conv[h:h_end,w:w_end]