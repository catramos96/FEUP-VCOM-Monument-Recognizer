#!/usr/bin/env python

from keras.preprocessing.image import load_img
from keras.preprocessing import image
import data_resources
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import cv2
from PIL import Image
import numpy as np
from keras.optimizers import Adam
from sklearn.utils import shuffle
from random import randint


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


# https: // github.com/qqwweee/keras-yolo3

def rgb_to_hsv(arr):
    """
    convert float rgb values (in the range [0, 1]), in a numpy array to hsv
    values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    hsv : (..., 3) ndarray
       Colors converted to hsv values in range [0, 1]
    """
    # make sure it is an ndarray
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {} was found.".format(arr.shape))

    in_ndim = arr.ndim
    if arr.ndim == 1:
        arr = np.array(arr, ndmin=2)

    # make sure we don't have an int image
    arr = arr.astype(np.promote_types(arr.dtype, np.float32))

    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = arr.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    if in_ndim == 1:
        out.shape = (3,)

    return out


# https: // github.com/qqwweee/keras-yolo3


def hsv_to_rgb(hsv):

    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    # if we got passed a 1D array, try to treat as
    # a single color and reshape as needed
    in_ndim = hsv.ndim
    if in_ndim == 1:
        hsv = np.array(hsv, ndmin=2)

    # make sure we don't have an int image
    hsv = hsv.astype(np.promote_types(hsv.dtype, np.float32))

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    # `np.stack([r, g, b], axis=-1)` (numpy 1.10).
    rgb = np.concatenate([r[..., None], g[..., None], b[..., None]], -1)

    if in_ndim == 1:
        rgb.shape = (3,)

    return rgb

# from: https://github.com/qqwweee/keras-yolo3S


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(','))))
                    for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]]*scale + dx
            box[:, [1, 3]] = box[:, [1, 3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter, 1+jitter)/rand(1-jitter, 1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1/rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
        box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


def draw_image_box(img, xmin, ymin, xmax, ymax):

    cv2.line(img, (xmin, ymin), (xmin, ymax), (255, 0, 0), 2)
    cv2.line(img, (xmin, ymin), (xmax, ymin), (255, 0, 0), 2)
    cv2.line(img, (xmax, ymax), (xmin, ymax), (255, 0, 0), 2)
    cv2.line(img, (xmax, ymax), (xmax, ymin), (255, 0, 0), 2)

    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_train_val_n_samples(validation_split):
    # calculate number of instances for training and validation
    min_inst = 1000000

    # min_inst = minimum instances of a class for balanced data
    for n_inst in data_resources.n_instances_info:
        if(n_inst < min_inst):
            min_inst = n_inst

    n_validation = round(min_inst * validation_split, 0)    # data * split
    n_training = min_inst - n_validation                    # data - n_valiidation

    return int(n_training), int(n_validation)


'''
If guarantee_class_division is True then the division of classes on the batches will be guaranteed.
For this the size of the batches will be n_classes * batches
'''


def generator(batch_size, input_shape, validation_split, is_training, guarantee_class_division):

    n_training, n_validation = get_train_val_n_samples(validation_split)

    while(True):

        X_batches = []
        Y_batches = []

        for b in range(batch_size):

            classes = []

            # choose random class per batch
            if(not guarantee_class_division):
                rnd_class = randint(0, len(data_resources.classes)-1)
                classes.append(rnd_class)
            # aditional batch per class
            else:
                classes = np.arange(len(data_resources.classes))

            for i in range(len(classes)):

                with open(data_resources.data_info[i]) as f:

                    content = f.readlines()
                    content = [x.strip() for x in content]

                    # choose random line
                    if(is_training):
                        rnd_line = randint(0, len(content[:n_training])-1)
                    else:
                        rnd_line = randint(
                            0, len(content[n_training:n_validation+n_training])-1)
                    line = content[rnd_line]

                    # random transformations
                    image, box = get_random_data(
                        line, input_shape, True, 1)

                    # just one box per image
                    box = box[0]

                    # build label
                    classes = np.zeros(5)
                    classes[int(box[4])] = 1

                    y = np.concatenate((classes, box[:4]))

                    # save to set
                    X_batches.append(image)
                    Y_batches.append(y)

        X_batches = np.array(X_batches)
        Y_batches = np.array(Y_batches)

        yield(X_batches, Y_batches)


def prepare_datasets(batch_size, image_size, validation_split):

    n_training, n_validation = get_train_val_n_samples(validation_split)

    total = n_validation + n_training

    X_training = []
    Y_training = []
    X_validation = []
    Y_validation = []

    n = 0
    batch = 0

    # TEST transformations on image and box
    '''
    text = "data/images/arrabida/arrabida-0037.jpg 0,160,593,292,1"

    image, box = get_random_data(text,(image_size,image_size),True,1)

    cv2.imshow('image', image)
    img = np.zeros((image_size,image_size))
    draw_image_box(image,int(box[0][0]),int(box[0][1]),int(box[0][2]),int(box[0][3]))
    print(box)
    '''

    # iterate classes informations .txt
    for info_path in data_resources.data_info:

        print("\n{}\n".format(info_path))

        with open(info_path) as f:
            content = f.readlines()
            content = [x.strip() for x in content]

            n = 0

            # iterate each line
            for line in content:

                if(n >= total):
                    break

                n += 1
                batch = 0

                # create batches per image
                for batch in range(0, batch_size):

                    batch += 1

                    image, box = get_random_data(
                        line, (image_size, image_size), True, 1)
                    box = box[0]  # just one box per image

                    classes = np.zeros(5)
                    classes[int(box[4])] = 1

                    y = np.concatenate((classes, box[:4]))

                    X_training.append(np.array(image))
                    Y_training.append(y)
                    print("OUTPUT ({}:{}): {}".format(n, batch, y))

    # shuffle
    X_training, Y_training = shuffle(X_training, Y_training, random_state=0)

    # size of training (training_images * batches * classes)
    size_t = n_training*batch_size*len(data_resources.classes)

    X_validation = X_training[size_t:]
    Y_validation = Y_training[size_t:]

    X_training = X_training[:size_t]
    Y_training = Y_training[:size_t]

    return X_training, X_validation, Y_training, Y_validation

# TEST
# X_t, X_v, Y_t, Y_v = prepare_datasets(3, 500, 0.3)


#data_generator(3, (500, 500), 0.3, True)
