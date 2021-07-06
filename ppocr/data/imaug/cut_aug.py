# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is based on https://github.com/zhunzhong07/Random-Erasing

import math
import random
import cv2

PI = math.pi
TEST_FLAG = False


def _show(img):
    if TEST_FLAG:
        cv2.imshow("iamge", img)
        cv2.waitKey(0)


"""
    随机cut某个区域
    args:
        target_area: 区域面积
        aspect_ratio: 长宽比
"""
def _random_erasing(img, x, y, h, w, target_area, aspect_ratio):
    for attempt in range(100):
        hh = int(round(math.sqrt(target_area * aspect_ratio)))
        ww = int(round(math.sqrt(target_area / aspect_ratio)))
        if ww < w and hh < h:
            x1 = random.randint(x, x + h - hh)
            y1 = random.randint(y, y + w - ww)
            img[x1:x1 + hh, y1:y1 + ww, :] = 0
            return img
    return img

"""
    随机cut某个区域
    args:
        sl: cut面积下限
        sh: cut面积上限
        r1: 长宽比率
"""
class RandomErasing(object):
    def __init__(self, sl=0.02, sh=0.4, r1=0.3):
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, *args):
        area = img.shape[0] * img.shape[1]
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, 1 / self.r1)
        img = _random_erasing(img, 0, 0, img.shape[0], img.shape[0], target_area, aspect_ratio)
        _show(img)
        return img

"""
    线性cut
    args:
        scale: 遮挡线的尺寸
        kl: 斜率下限
        kh: 斜率上限
"""
class LineErasing(object):
    def __init__(self, scale=0.05, kl=-PI / 4, kh=PI / 4):
        self.kl = kl
        self.kh = kh
        self.scale = scale

    def __call__(self, img, *args):
        h = img.shape[0]
        w = img.shape[1]
        k = random.uniform(self.kl, self.kh)
        if h > w:
            k = (PI / 2 - k) if k > 0 else (-PI / 2 - k)

        self.size = math.ceil(w * self.scale if h > w else h * self.scale)
        cx = h // 2
        cy = w // 2
        offset = math.ceil(cx / math.tan(k))

        cv2.line(img, (cy + offset, 0), (cy - offset, h), (0, 0, 0), self.size)
        _show(img)
        return img


"""
    按字符随机cut某个区域
    args:
        sl: cut面积下限
        sh: cut面积上限
        r1: 长宽比率
"""
class SingleErasing(object):
    def __init__(self, sl=0.1, sh=0.2, r1=1):
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, num, *args):
        h, w = img.shape[0], img.shape[1]
        sw, sh = w, h
        if w > h:
            sw = w // num
        else:
            sh = h // num

        x, y = 0, 0
        area = sh * sw

        for i in range(num):
            xx, yy = x, y
            if w > h:
                yy = y + i * sw
            else:
                xx = x + i * sh
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            img = _random_erasing(img, xx, yy, sh, sw, target_area, aspect_ratio)

        _show(img)
        return img


"""
    按字符左右cut
    args:
        sl: cut面积下限
        sh: cut面积上限
        prob：cut概率
"""
class CropErasing(object):
    def __init__(self, sl=0.2, sh=0.6, prob=0.5):
        self.sl = sl
        self.sh = sh
        self.prob = prob

    def __call__(self, img, num, *args):
        h, w = img.shape[0], img.shape[1]
        sw, sh = w, h
        if w > h:
            sw = w // num
        else:
            sh = h // num

        x, y = 0, 0

        for i in range(num):
            if random.random() <= self.prob:
                continue
            xx, yy = x, y
            if w > h:
                yy = y + i * sw
            else:
                xx = x + i * sh
            r = random.uniform(self.sl, self.sh)
            img[xx:, yy:yy + math.ceil(r * sw), :] = 0

        _show(img)
        return img
