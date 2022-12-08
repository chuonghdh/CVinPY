import numpy as np

    # To convert an RGBA image array that represents data with floating
    # point numbers from 0 to 1 into the RGB integer format from 0 to 255,
    # we need to make 4 changes:

    # 1) Get rid of the A channel
    # 2) Multiply by 255
    # 3) Round the resulting values
    # 4) Ensure values are between 0 and 255
    # 5) Convert data to 8-bit integers


def adapt_PNG(the_PNG):
    the_PNG = the_PNG[:, :, :3]
    the_PNG = the_PNG * 255
    the_PNG = adapt_image(the_PNG)
    return the_PNG


def adapt_image(the_img):
    return np.uint8(np.clip(the_img.round(), 0, 255))  # step 3, 4, 5

def RGB_to_grayscale(RGB_pic):
    rows, cols, temp = np.shape(RGB_pic)
    gs = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            gs[i, j] = np.average(RGB_pic[i, j])
    return gs

def grayscale_to_BW(grayscale_pic, threshold):
    rows, cols = np.shape(grayscale_pic)
    BW_pic = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            BW_pic[i, j] = 0 if grayscale_pic[i, j] <= threshold else 255
    return BW_pic