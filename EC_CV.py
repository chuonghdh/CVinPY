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

