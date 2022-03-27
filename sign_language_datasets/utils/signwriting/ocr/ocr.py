"""
This SignWriting OCR code is adapted from
https://colab.research.google.com/drive/1x0OupzZNkQW1rCiDjEe1LX5V9k8_YF69#scrollTo=_1YX_pILnjFe
"""
import functools
from typing import List

import numpy as np
from numpy.lib.stride_tricks import as_strided
from cv2 import cv2
from PIL import Image, ImageDraw, ImageFont
import os


def strided_convolution(image, weight, stride=1):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
    out_strides = (image.strides[0] * stride, image.strides[1] * stride, image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))


def rgb2bin(img, neg=0.1):
    return np.where(img[:, :, 0] < 200, 1, neg)


def key2swu(key):
    return chr(0x40001 +
               ((int(key[1:4], 16) - 256) * 96) +
               ((int(key[4:5], 16)) * 16) +
               int(key[5:6], 16))


def shape_pos(shape):
    x, y = shape[:2]
    return f'{x:03}x{y:03}'


def crop_whitespace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    if y > 0:
        y -= 1
        h += 1
    if x > 0:
        x -= 1
        w += 1
    if y + h < img.shape[0]:
        h += 1
    if x + w < img.shape[1]:
        w += 1

    return cv2.copyMakeBorder(img[y:y + h, x:x + w], 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))


@functools.lru_cache()
def get_font():
    dirname = os.path.dirname(__file__)
    font_path = os.path.join(dirname, "assets/SuttonSignWritingOneD.ttf")

    return ImageFont.truetype(font_path, 30)


def image_to_fsw(image: np.ndarray, symbols: List[str]) -> str:
    font = get_font()

    # Adding border for conv calc to go over borders
    img_rgb = cv2.copyMakeBorder(image.copy(), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img_bin = rgb2bin(img_rgb, -0.5)

    img_width, img_height, _ = img_rgb.shape
    final_sign = "M" + shape_pos([500, 500])
    # final_sign = "A" + "".join(symbols) + "M" + shape_pos(img_rgb.shape)

    # Create all convolutions
    templates = []
    convs = []
    for symbol in symbols:
        swu = key2swu(symbol)
        (width, height), (offset_x, offset_y) = font.font.getsize(swu)
        img = Image.new("RGB", (width - offset_x, height - offset_y), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((-offset_x, -offset_y), swu, font=font, fill=(0, 0, 0))

        template = crop_whitespace(np.array(img))
        templates.append(template)

        conv = strided_convolution(img_bin, rgb2bin(template, -0.5))
        convs.append(conv)

    # Save video?
    # out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, img_rgb.shape[:2])

    # Find max symbol
    while len(symbols) > 0:
        # 1. Find best match
        best_match = {"symbol": "", "point": (), "score": 0}
        for symbol, conv in zip(symbols, convs):
            point = np.unravel_index(np.argmax(conv, axis=None), conv.shape)
            score = conv[point]
            if score > best_match["score"]:
                best_match = {"symbol": symbol, "point": point, "score": score}

        idx = symbols.index(best_match["symbol"])

        # 2. Select best match
        pt = best_match["point"][::-1]
        position = [int(500 - img_width/2) + pt[0], int(500 - img_height/2) + pt[1]]
        final_sign += best_match["symbol"] + shape_pos(position)

        w, h = templates[idx].shape[:-1]
        cv2.rectangle(img_rgb, pt, (pt[0] + h, pt[1] + w), (0, 0, 255), 1)

        # 3. Remove symbol from list
        # print(best_match)
        # cv2_imshow(templates[idx])
        # cv2_imshow(convs[idx])
        # out.write(img_rgb)
        symbols.pop(idx)
        convs.pop(idx)
        templates.pop(idx)

        # 4. Prevent collisions with this match
        for conv in convs:
            x, y = pt
            conv[y - 2:y + 4, x - 2:x + 4] = 0

    return final_sign


if __name__ == "__main__":
    img_rgb = cv2.imread('assets/sign.png')
    symbols = ['S1f520', 'S1f528', 'S23c04', 'S23c1c', 'S2fb04', 'S2ff00', 'S33b10']

    fsw = image_to_fsw(img_rgb, symbols)
    print("fsw", fsw)
