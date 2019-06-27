#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.



"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

labelPath1 = 'label_data_train.txt'
f_labeltxt = open(labelPath1,'w')
labelPath2 = 'label_data_test.txt'
f_labeltxt2 = open(labelPath2,'w')
FONT_DIR = "./fonts"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (128, 128)

CHARS = common.CHARS + " "


def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(font_path, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]#row
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape, 
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale or scale < min_scale:
        out_of_bounds = True
    roll = random.uniform(-0.3, 0.3) * rotation_variation
    pitch = random.uniform(-0.2, 0.2) * rotation_variation
    yaw = random.uniform(-1.2, 1.2) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])


    return M, out_of_bounds


def generate_code():
    #ormat={"{}{}{}{} {}{}{}","{}{}{}{}{}{}{}"}
    input = []
    for i in range(7):
        if random.randint(0,1)==0:
            input.append( random.choice(common.LETTERS))
        # elif random.randint(0,2)==1:
        #     input.append(random.choice(common.DIGITS))
        else:
            input.append(random.choice(common.DIGITS))
    return "{}{} {}{}{}{}".format(input[0],input[1],input[2],input[3],input[4],input[5],input[6])

    # return "{}{}{}{} {}{}{}".format(
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.DIGITS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS),
    #     random.choice(common.LETTERS))


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padding
    count=0
    letters=[]
    for c in code:
        count+=1
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

        #letter
        if c!=' ':
            letter = numpy.zeros(out_shape)
            letter[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
            letter = letter * 255
            # cv2.imwrite(c+'mask.jpg',letter)
            letters.append(letter)
            # letter1 = text_mask*255
    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", ""),letters


def generate_bg(num_bg_images):
    found = False
    length = len(num_bg_images)
    while not found:
        index = int(random.randint(0, length - 1))
        img_name = num_bg_images[index]
        fname = "bgs/"+img_name
        # bg = cv2.imread(fname)
        bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    # x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    # y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    # bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]
    bg = cv2.resize(bg,(OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]))

    return bg,img_name


def generate_im(char_ims, num_bg_images):
    bg,img_name = generate_bg(num_bg_images)

    plate, plate_mask, code,letters = generate_plate(FONT_HEIGHT, char_ims)
    
    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.78,
                            max_scale=0.875,
                            rotation_variation=1.0,
                            scale_variation=1.5,
                            translation_variation=1.0)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))

    letters_Affine=[]
    index = 0
    label_txt=""
    for (c,letter) in zip(code,letters):
        # label_txt +=' '
        letter = cv2.warpAffine(letter, M, (bg.shape[1], bg.shape[0]))
        # cv2.imwrite('_'+c+'.jpg', letters_Affine[index])
        s = writeContour(letter,c)
        # if s=="": #
        label_txt+= s
        index+=1
    label_txt += '\n'
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    # cv2.imwrite('mask/'+img_name, plate_mask* 255.)

    out = plate * plate_mask + bg * (1.0 - plate_mask)
    cv2.imwrite('plate.jpg', plate*255)
    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)
    # cv2.imwrite('plate.jpg', out* 255.)
    return plate_mask,out, code, not out_of_bounds ,label_txt


def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


def generate_ims():
    """
    Generate number plate images.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    fonts, font_char_ims = load_fonts(FONT_DIR)
    num_bg_images = os.listdir("bgs")
    while True:
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)

def writeContour(letter,curClass):
    # a = cv2.imread(img_path + '/' + img_name)
    from skimage import measure, data, color
    # import cv2

    img = color.rgb2gray(letter)
    # if img_name =='model1444.jpg':
    #     a=0
    # img=color.rgb2gray(data.horse())

    # 检测所有图形的轮廓
    contours = measure.find_contours(img, 0.5)
    if len(contours) < 1:
        return ""
    edge = contours[0]

    # write txt
    # print(img_name)
    # curClass = img_name[9]
    output_txt =curClass + ' '

    pts_num = 0
    for i in range(len(edge)):
        if i % 5 == 0:
            pts_num += 1
    print('pts_num:', pts_num)
    output_txt +=str(pts_num)
    output_txt +=' '
    t1 = 0
    t2 = 0
    for i in range(len(edge)):
        if i % 5 == 0:
            output_txt +=str(int(edge[i, 1]))
            output_txt +=' '
            t1 += 1
    # print('t1:',t1)
    for i in range(len(edge)):
        if i % 5 == 0:
            output_txt +=str(int(edge[i, 0]))
            output_txt +=' '
            t2 += 1

    return output_txt
    

if __name__ == "__main__":
    if not os.path.exists('train_Bigscale'):
        os.mkdir("train_Bigscale")
    if not os.path.exists('test_Bigscale'):
        os.mkdir("test_Bigscale")

    im_gen = itertools.islice(generate_ims(), 100000)
    for img_idx, (plate_mask, im, c, p, label_txt) in enumerate(im_gen):
        fname = "train_Bigscale/{:08d}_{}_{}.png".format(img_idx, c,
                                                        "1" if p else "0")
        img_name = "{:08d}_{}_{}.png".format(img_idx, c,
                                             "1" if p else "0")
        fname2 = "mask_Bigscale/{:08d}_{}_{}.png".format(img_idx, c,
                                                         "1" if p else "0")
        print(str(img_idx) + "/100000---------"+fname)
        cv2.imwrite(fname, im * 255.)
        f_labeltxt.write(img_name + ' 6 ' + label_txt)
        # print(img_idx+"/100000")

    im_gen = itertools.islice(generate_ims(), 20000)
    for img_idx, (plate_mask, im, c, p, label_txt) in enumerate(im_gen):
        fname = "test_Bigscale/{:08d}_{}_{}.png".format(img_idx, c,
                                                        "1" if p else "0")
        img_name = "{:08d}_{}_{}.png".format(img_idx, c,
                                             "1" if p else "0")
        fname2 = "mask_Bigscale/{:08d}_{}_{}.png".format(img_idx, c,
                                                         "1" if p else "0")
        print(str(img_idx)+"/20000---------"+fname)
        cv2.imwrite(fname, im * 255.)
        f_labeltxt2.write(img_name + ' 6 ' + label_txt)
        # cv2.imwrite(fname2, plate_mask * 255.)

f_labeltxt.close()
f_labeltxt2.close()