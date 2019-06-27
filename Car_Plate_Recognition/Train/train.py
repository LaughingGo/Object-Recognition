
"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import json
import skimage
from mrcnn import visualize
import cv2
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".


import zipfile
import urllib.request
import shutil
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################
command='test'
# command='train'
args = {'command':command,
            'dataset':'/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data',
            'model':'mask_rcnn_coco.h5',
            'logs':'logs',
            'year':'2017'
            }

class PLCConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "PLC"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 36  # COCO has 80 classes

    STEPS_PER_EPOCH = 100

    # base net, resnet101 or resnet50
    BACKBONE = 'resnet50'
    LEARNING_RATE = 0.001
    # ROIs kept after non-maximum suppression (training and inference) -X
    # POST_NMS_ROIS_INFERENCE = 250

    # # Size of the fully-connected layers in the classification graph  -X
    # FPN_CLASSIF_FC_LAYERS_SIZE = 512
    # #
    # # # Size of the top-down layers used to build the feature pyramid  -?
    # TOP_DOWN_PYRAMID_SIZE = 128

    # Input image resizing -work! -Inference
    # IMAGE_MIN_DIM = int(400)
    # IMAGE_MAX_DIM = int(512)

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    # BACKBONE_STRIDES = [8, 16, 32, 64,128]
from threading import Thread, Lock

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

############################################################
#  Dataset--PLC
############################################################

class PLCDataset(utils.Dataset):

    def load_PLC(self, dataset_dir, subset):
        """Load a subset of the PLC dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have five class to add.
        self.add_class("PLC", 1, "A")
        # self.add_class("PLC", 2, "B")
        # self.add_class("PLC", 3, "C")
        # self.add_class("PLC", 4, "D")
        # self.add_class("PLC", 5, "E")


        # Train or validation dataset?
        assert subset in ["json_train", "json_val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # print('regions：',a['regions'].values)
            polygons = [a['regions'][r]['shape_attributes'] for r in range(len(a['regions']))]
            # polygons = [r['shape_attributes'] for r in a['regions'].values()]
            names = [a['regions'][r]['region_attributes']['name'] for r in range(len(a['regions']))]
            # 序列字典
            name_dict = {"A": 1}
            name_id = [name_dict[n] for n in names]


            # for index in range(len(polygons)):
            #     polygons[index]['name'] = a['regions'][index]['region_attributes']['name']
            # a['regions'][0]['region_attributes']['name'] = a['regions'][0]['region_attributes']['name']

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "PLC",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                class_id = name_id,
                polygons=polygons
                )

    def load_PLC_AutoLable(self, dataset_dir, subset):
        """Load a subset of the PLC dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have five class to add.
        # self.add_class("PLC", 1, "7")
        # self.add_class("PLC", 2, "R")

        self.add_class("PLC", 1, "0")
        self.add_class("PLC", 2, "1")
        self.add_class("PLC", 3, "2")
        self.add_class("PLC", 4, "3")
        self.add_class("PLC", 5, "4")
        self.add_class("PLC", 6, "5")
        self.add_class("PLC", 7, "6")
        self.add_class("PLC", 8, "7")
        self.add_class("PLC", 9, "8")
        self.add_class("PLC", 10, "9")
        self.add_class("PLC", 11, "A")
        self.add_class("PLC", 12, "B")
        self.add_class("PLC", 13, "C")
        self.add_class("PLC", 14, "D")
        self.add_class("PLC", 15, "E")
        self.add_class("PLC", 16, "F")
        self.add_class("PLC", 17, "G")
        self.add_class("PLC", 18, "H")
        self.add_class("PLC", 19, "I")
        self.add_class("PLC", 20, "J")
        self.add_class("PLC", 21, "K")
        self.add_class("PLC", 22, "L")
        self.add_class("PLC", 23, "M")
        self.add_class("PLC", 24, "N")
        self.add_class("PLC", 25, "O")
        self.add_class("PLC", 26, "P")
        self.add_class("PLC", 27, "Q")
        self.add_class("PLC", 28, "R")
        self.add_class("PLC", 29, "S")
        self.add_class("PLC", 30, "T")
        self.add_class("PLC", 31, "U")
        self.add_class("PLC", 32, "V")
        self.add_class("PLC", 33, "W")
        self.add_class("PLC", 34, "X")
        self.add_class("PLC", 35, "Y")
        self.add_class("PLC", 36, "Z")


        # self.add_class("PLC", 3, "C")
        # self.add_class("PLC", 4, "D")
        # self.add_class("PLC", 5, "E")


        # Train or validation dataset?
        # assert subset in ["json_train2", "json_val2"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        inputjson_f = open(os.path.join(dataset_dir, 'label_data.txt'), 'r')
        line = inputjson_f.readline()
        if subset =="json_val_carplate":
            line = inputjson_f.readline()
        while line != "":
            tag=1
            lines_items = line.split(' ')
            img_name = lines_items[0]
            obj_num = lines_items[1]
            obj_start_index = 2
            obj_pts = 0
            polygons = []
            names = []
            for i in range(int(obj_num)):
                polygon = {}
                polygon['name'] = 'polygon'
                obj_name = lines_items[obj_start_index]
                #### 2 class
                if len(img_name)==9 and int(img_name[5])>2:
                    obj_name="B"
                    print("A--->B",img_name)
                names.append(obj_name)
                obj_pts = int(lines_items[obj_start_index + 1])
                polygon['all_points_x'] = [int(lines_items[obj_start_index + 1 + j]) for j in range(1, obj_pts + 1)]
                polygon['all_points_y'] = [int(lines_items[obj_start_index + 1 + j]) for j in range(obj_pts + 1, obj_pts * 2 + 1)]
                for j in range(obj_pts + 1, obj_pts * 2 + 1):
                    if(int(lines_items[obj_start_index + 1 + j])>=720):
                        tag=0
                        break

                polygons.append(polygon)
                # print('img:', img_name, 'obj:', obj_name, 'obj pts:',
                #       [int(lines_items[obj_start_index + 1 + j]) for j in range(1, obj_pts * 2 + 1)])
                obj_start_index += obj_pts * 2 + 2
            line = inputjson_f.readline()
            if subset == "json_val_carplate":
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
                line = inputjson_f.readline()
            # name_dict = {"7": 1,"R":2}

            name_dict = {"0": 1,
                         "1": 2,
                         "2": 3,
                         "3": 4,
                         "4": 5,
                         "5": 6,
                         "6": 7,
                         "7": 8,
                         "8": 9,
                         "9": 10,
                         "A": 11,
                         "B": 12,
                         "C": 13,
                         "D": 14,
                         "E": 15,
                         "F": 16,
                         "G": 17,
                         "H": 18,
                         "I": 19,
                         "J": 20,
                         "K": 21,
                         "L": 22,
                         "M": 23,
                         "N": 24,
                         "O": 25,
                         "P": 26,
                         "Q": 27,
                         "R": 28,
                         "S": 29,
                         "T": 30,
                         "U": 31,
                         "V": 32,
                         "W": 33,
                         "X": 34,
                         "Y": 35,
                         "Z": 36

                         }
            name_id = [name_dict[n] for n in names]


        # for a in annotations:
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. There are stores in the
        #     # shape_attributes (see json format above)
        #     # print('regions：',a['regions'].values)
        #     polygons = [a['regions'][r]['shape_attributes'] for r in range(len(a['regions']))]
        #     # polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     names = [a['regions'][r]['region_attributes']['name'] for r in range(len(a['regions']))]
        #     # 序列字典
        #     name_dict = {"A": 1, "B": 2, "C": 3,"D":4,"E":5}
        #     name_id = [name_dict[n] for n in names]
        #
        #
        #     # for index in range(len(polygons)):
        #     #     polygons[index]['name'] = a['regions'][index]['region_attributes']['name']
        #     # a['regions'][0]['region_attributes']['name'] = a['regions'][0]['region_attributes']['name']
        #
        #     # load_mask() needs the image size to convert polygons to masks.
        #     # Unfortunately, VIA doesn't include it in JSON, so we must read
        #     # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, img_name)
            #########
            image_path+='.jpg'
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            if tag==1:
                self.add_image(
                    "PLC",
                    image_id=img_name,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    class_id = name_id,
                    polygons=polygons
                    )
                print('img:', img_name)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "PLC":
            return super(self.__class__, self).load_mask(image_id)

        name_id = image_info["class_id"]
        print(name_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.array(name_id, dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if 'all_points_y' in p.keys():
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            elif 'width' in p.keys():
                rr, cc = skimage.draw.polygon([p['y'],p['y'],p['y']+p['height'],p['height']],[p['x'],p['x']+p['width'],p['x']+p['width'],p['x']])
            mask[rr, cc, i] = 1

        # print( mask.astype(np.bool), name_id)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return (mask.astype(np.bool), class_ids)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "PLC":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PLCDataset()
    # dataset_train.load_PLC_AutoLable(args['dataset'], "json_train2")
    dataset_train.load_PLC_AutoLable(args['dataset'], "json_train_carplate")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PLCDataset()
    # dataset_val.load_PLC(args['dataset'], "json_val")
    dataset_val.load_PLC_AutoLable(args['dataset'], "json_val_carplate")

    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='heads')

#
# ############################################################
# #  Dataset
# ############################################################
#
# class PLCDataset(utils.Dataset):
#     def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
#                   class_map=None, return_coco=False, auto_download=False):
#         """Load a subset of the COCO dataset.
#         dataset_dir: The root directory of the COCO dataset.
#         subset: What to load (train, val, minival, valminusminival)
#         year: What dataset year to load (2014, 2017) as a string, not an integer
#         class_ids: If provided, only loads images that have the given classes.
#         class_map: TODO: Not implemented yet. Supports maping classes from
#             different datasets to the same class ID.
#         return_coco: If True, returns the COCO object.
#         auto_download: Automatically download and unzip MS-COCO images and annotations
#         """
#
#         if auto_download is True:
#             self.auto_download(dataset_dir, subset, year)
#
#         coco = COCO("{}/PLC_data/annotations/instances_{}.json".format(dataset_dir, subset))
#         # coco = COCO(dataset_dir)
#         if subset == "minival" or subset == "valminusminival":
#             subset = "val"
#         image_dir = "{}/{}".format(dataset_dir, subset)
#
#         # Load all classes or a subset?
#         if not class_ids:
#             # All classes
#             class_ids = sorted(coco.getCatIds())
#
#         # All images or a subset?
#         if class_ids:
#             image_ids = []
#             for id in class_ids:
#                 image_ids.extend(list(coco.getImgIds(catIds=[id])))
#             # Remove duplicates
#             image_ids = list(set(image_ids))
#         else:
#             # All images
#             image_ids = list(coco.imgs.keys())
#
#         # Add classes
#         for i in class_ids:
#             self.add_class("coco", i, coco.loadCats(i)[0]["name"])
#
#         # Add images
#         for i in image_ids:
#             self.add_image(
#                 "coco", image_id=i,
#                 path=os.path.join(image_dir, coco.imgs[i]['file_name']),
#                 width=coco.imgs[i]["width"],
#                 height=coco.imgs[i]["height"],
#                 annotations=coco.loadAnns(coco.getAnnIds(
#                     imgIds=[i], catIds=class_ids, iscrowd=None)))
#         if return_coco:
#             return coco
#
#     def auto_download(self, dataDir, dataType, dataYear):
#         """Download the COCO dataset/annotations if requested.
#         dataDir: The root directory of the COCO dataset.
#         dataType: What to load (train, val, minival, valminusminival)
#         dataYear: What dataset year to load (2014, 2017) as a string, not an integer
#         Note:
#             For 2014, use "train", "val", "minival", or "valminusminival"
#             For 2017, only "train" and "val" annotations are available
#         """
#
#         # Setup paths and file names
#         if dataType == "minival" or dataType == "valminusminival":
#             imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
#             imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
#             imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
#         else:
#             imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
#             imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
#             imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
#         # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)
#
#         # Create main folder if it doesn't exist yet
#         if not os.path.exists(dataDir):
#             os.makedirs(dataDir)
#
#         # Download images if not available locally
#         if not os.path.exists(imgDir):
#             os.makedirs(imgDir)
#             print("Downloading images to " + imgZipFile + " ...")
#             with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
#                 shutil.copyfileobj(resp, out)
#             print("... done downloading.")
#             print("Unzipping " + imgZipFile)
#             with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
#                 zip_ref.extractall(dataDir)
#             print("... done unzipping")
#         print("Will use images in " + imgDir)
#
#         # Setup annotations data paths
#         annDir = "{}/annotations".format(dataDir)
#         if dataType == "minival":
#             annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
#             annFile = "{}/instances_minival2014.json".format(annDir)
#             annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
#             unZipDir = annDir
#         elif dataType == "valminusminival":
#             annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
#             annFile = "{}/instances_valminusminival2014.json".format(annDir)
#             annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
#             unZipDir = annDir
#         else:
#             annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
#             annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
#             annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
#             unZipDir = dataDir
#         # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)
#
#         # Download annotations if not available locally
#         if not os.path.exists(annDir):
#             os.makedirs(annDir)
#         if not os.path.exists(annFile):
#             if not os.path.exists(annZipFile):
#                 print("Downloading zipped annotations to " + annZipFile + " ...")
#                 with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
#                     shutil.copyfileobj(resp, out)
#                 print("... done downloading.")
#             print("Unzipping " + annZipFile)
#             with zipfile.ZipFile(annZipFile, "r") as zip_ref:
#                 zip_ref.extractall(unZipDir)
#             print("... done unzipping")
#         print("Will use annotations in " + annFile)
#
#     def load_mask(self, image_id):
#         """Load instance masks for the given image.
#
#         Different datasets use different ways to store masks. This
#         function converts the different mask format to one format
#         in the form of a bitmap [height, width, instances].
#
#         Returns:
#         masks: A bool array of shape [height, width, instance count] with
#             one mask per instance.
#         class_ids: a 1D array of class IDs of the instance masks.
#         """
#         # If not a COCO image, delegate to parent class.
#         image_info = self.image_info[image_id]
#         if image_info["source"] != "coco":
#             return super(CocoDataset, self).load_mask(image_id)
#
#         instance_masks = []
#         class_ids = []
#         annotations = self.image_info[image_id]["annotations"]
#         # Build mask of shape [height, width, instance_count] and list
#         # of class IDs that correspond to each channel of the mask.
#         for annotation in annotations:
#             class_id = self.map_source_class_id(
#                 "coco.{}".format(annotation['category_id']))
#             if class_id:
#                 m = self.annToMask(annotation, image_info["height"],
#                                    image_info["width"])
#                 # Some objects are so small that they're less than 1 pixel area
#                 # and end up rounded out. Skip those objects.
#                 if m.max() < 1:
#                     continue
#                 # Is it a crowd? If so, use a negative class ID.
#                 if annotation['iscrowd']:
#                     # Use negative class ID for crowds
#                     class_id *= -1
#                     # For crowd masks, annToMask() sometimes returns a mask
#                     # smaller than the given dimensions. If so, resize it.
#                     if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
#                         m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
#                 instance_masks.append(m)
#                 class_ids.append(class_id)
#
#         # Pack instance masks into an array
#         if class_ids:
#             mask = np.stack(instance_masks, axis=2).astype(np.bool)
#             class_ids = np.array(class_ids, dtype=np.int32)
#             return mask, class_ids
#         else:
#             # Call super class to return an empty mask
#             return super(CocoDataset, self).load_mask(image_id)
#
#     def image_reference(self, image_id):
#         """Return a link to the image in the COCO Website."""
#         info = self.image_info[image_id]
#         if info["source"] == "coco":
#             return "http://cocodataset.org/#explore?id={}".format(info["id"])
#         else:
#             super(CocoDataset, self).image_reference(image_id)
#
#     # The following two functions are from pycocotools with a few changes.
#
#     def annToRLE(self, ann, height, width):
#         """
#         Convert annotation which can be polygons, uncompressed RLE to RLE.
#         :return: binary mask (numpy 2D array)
#         """
#         segm = ann['segmentation']
#         if isinstance(segm, list):
#             # polygon -- a single object might consist of multiple parts
#             # we merge all parts into one mask rle code
#             rles = maskUtils.frPyObjects(segm, height, width)
#             rle = maskUtils.merge(rles)
#         elif isinstance(segm['counts'], list):
#             # uncompressed RLE
#             rle = maskUtils.frPyObjects(segm, height, width)
#         else:
#             # rle
#             rle = ann['segmentation']
#         return rle
#
#     def annToMask(self, ann, height, width):
#         """
#         Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
#         :return: binary mask (numpy 2D array)
#         """
#         rle = self.annToRLE(ann, height, width)
#         m = maskUtils.decode(rle)
#         return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

def display_mask_image(cv_window_name,image, boxes, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                       score_threshold=0.8):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
        return image
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = True
    # if not ax:
    #     _, ax = plt.subplots(1, figsize=figsize)
    #     auto_show = True
    #
    # Generate random colors
    # colors = visualize.fixed_colors(N)
    colors = visualize.get_colors(N)
    # Show area outside image boundaries.
    # height, width = image.shape[:2]
    # ax.set_ylim(height + 10, -10)
    # ax.set_xlim(-10, width + 10)
    # ax.axis('off')
    # ax.set_title(title)
    scoreMin=score_threshold
    masked_image=image.astype(np.uint8).copy()
    for i in range(N):
        print('class:',class_names[class_ids[i]],'score:',scores[i])
        if scores[i] > scoreMin:
            # Mask
            # mask = masks[:, :, i]
            # color = colors[i]
            # color = colors[class_ids[i]]
            color = colors[0]
            # color =
            # if show_mask:
                # masked_image = visualize.apply_mask(image, mask, color,alpha=0.4)

    masked_image = masked_image.astype(np.uint8).copy()
    for i in range(N):
        if scores[i]>scoreMin:
            # color = colors[i]
            # color = colors[class_ids[i]]
            color = colors[0]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                image = cv2.rectangle(masked_image, (x1, y1),(x2, y2),(100,20,100),thickness=2)
                # p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                #                     alpha=0.7, linestyle="dashed",
                #                     edgecolor=color, facecolor='none')
                # ax.add_patch(p)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                # x = random.randint(x1, (x1 + x2) // 2)
                # caption = "{}  {:.3f}".format(label, score) if score else label
                caption = "{}  ".format(label) if score else label
            else:
                caption = captions[i]

            image = cv2.addText(image,caption, (x1, y1-1),'Times',pointSize=13,color=(178,34,34))
            # ax.text(x1, y1 + 8, caption,
            #         color='w', size=11, backgroundcolor="none")


            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            # padded_mask = np.zeros(
            #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            # padded_mask[1:-1, 1:-1] = mask
            # contours = visualize.find_contours(padded_mask, 0.5)
            # for verts in contours:
            #     # Subtract the padding and flip (y, x) to (x, y)
            #     verts = np.fliplr(verts) - 1
            #     print('verts:',verts)
                # points = np.array([[(1,1),(20,20),(15,15)]], dtype=np.int32)
                # masked_image = cv2.fillPoly(masked_image,points,color)
                # p = Polygon(verts, facecolor="none", edgecolor=color)
                # ax.add_patch(p)
        # ax.imshow(masked_image.astype(np.uint8))
        # cv2.imshow(cv_window_name, masked_image)
        # cv2.waitKey(1)
        # if auto_show:
        #     plt.show()
    return masked_image



def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)



def preprocess_image(source_image,width,height):
    resized_image = cv2.resize(source_image, (int(width/4), int(height/4)))
    # trasnform values from range 0-255 to range -1.0 - 1.0
    # resized_image = resized_image - 127.5
    # resized_image = resized_image * 0.007843
    return resized_image


def detect(model, image_path=None, video_path=None,image_dir = None,camera=None,Min_score = 0.5,watikey=True):
    assert image_path or video_path or image_dir or camera
    if image_dir:
        # write_img=1
        write_img = 0
        imgs = os.listdir(image_dir)
        for img in imgs:
            # Run model detection and generate the color splash effect
            print("Running on {}".format(img))
            # Read image
            if image_dir=='/home/jianfenghuang/Myproject/CarPlateRecog/data/Number_Letter/numbers/number_letter_test':
                image = skimage.io.imread(image_dir + '/' + img)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = skimage.io.imread(image_dir+'/'+img,cv2.IMREAD_GRAYSCALE)

            # OpenCV returns images as BGR, convert to RGB
            # image = skimage.io.imread(image_dir + '/' + img)
            image = image[..., ::-1]
            # image = image[..., ::-1]
            # image = image[..., ::-1]
            # Detect objects
            t1 = time.time()
            r = model.detect([image], verbose=1)[0]
            print('detect time: ', time.time() - t1)
            cv_window_name = "Mask-RCNN for PLC"
            cv2.namedWindow(cv_window_name)
            # cv2.moveWindow(cv_window_name, 10, 10)
            # class_names = ['BG', '7','R']
            class_names = ['BG','0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                           'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            # print('redult:',list(r))
            mask_img = display_mask_image(cv_window_name, image, r['rois'], r['class_ids'],
                                          class_names, r['scores'], show_bbox=True,score_threshold=Min_score,show_mask=False)
            # print('PLC_data/masked_p1/masked_'+img[:-4])
            if write_img==1:
                cv2.imwrite('PLC_data/masked_p1/masked_'+img[:-4]+'.jpg',mask_img)
            # foutput.writelines(img + ':\n')
            str = r['rois']
            np.savetxt('PLC_data/masked_txt/'+img[:-4]+'.txt',str, fmt='%.2f')

            # pic = visualize.m_apply_mask(image,r['masks'])
            if write_img == 1:
                arr = np.zeros([720, 1280])

                for index in range(r['masks'].shape[2]):
                    if r['scores'][index] > 0.8:
                        mask = r['masks'][:, :, index].reshape([720, 1280])
                        arr[mask] = 255
                cv2.imwrite('PLC_data/bin_map/'+img[:-4]+'.jpg',arr)
            # np.savetxt('test.txt', a, fmt='%.2f')
            # foutput.writelines('\n')
            cv2.imshow(cv_window_name, mask_img)
            if watikey:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
        # foutput.close()
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # # OpenCV returns images as BGR, convert to RGB
        # image = image[..., ::-1]
        # Detect objects
        t1 = time.time()
        r = model.detect([image], verbose=1)[0]
        print('detect time: ', time.time() - t1)
        cv_window_name = "Mask-RCNN for PLC"
        cv2.namedWindow(cv_window_name)
        # cv2.moveWindow(cv_window_name, 10, 10)
        class_names = ['BG', 'Simatic IOT2000', 'Raspberry Pi', 'Nuc', 'DR-120-24', 'RS30']
        mask_img = display_mask_image(cv_window_name, image, r['rois'], r['class_ids'],
                                      class_names, r['scores'])

        cv2.imshow(cv_window_name, mask_img)
        cv2.waitKey(0)
        # Color splash
        # splash = color_splash(image, r['masks'])
        # # Save output
        # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        # skimage.io.imsave(file_name, splash)
    elif video_path:

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        # file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        # vwriter = cv2.VideoWriter(file_name,
        #                           cv2.VideoWriter_fourcc(*'MJPG'),
        #                           fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()

            if success:
                image = preprocess_image(image, width, height)
                # OpenCV returns images as BGR, convert to RGB
                # image = image[..., ::-1]
                # Detect objects
                t1 = time.time()
                r = model.detect([image], verbose=0)[0]
                print('detect time: ', time.time() - t1)
                cv_window_name = "Mask-RCNN for PLC"
                cv2.namedWindow(cv_window_name)
                # cv2.moveWindow(cv_window_name, 10, 10)
                class_names = ['BG', 'Simatic IOT2000', 'Raspberry Pi', 'Nuc', 'DR-120-24', 'RS30']
                # print('redult:',list(r))
                mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
                                                       class_names, r['scores'],score_threshold=0.9)

                cv2.imshow(cv_window_name, mask_img)
                cv2.waitKey(1)
                count += 1
            else:
                print("video stop!")
                break
        # vwriter.release()
        cv2.destroyAllWindows()
    if camera:
        count = 1
        while (True):
            try:
                image = camera.read()
                # OpenCV returns images as BGR, convert to RGB
                # image = image[..., ::-1]
                # image = preprocess_image(image, 720, 1280)
                t1 = time.time()
                r = model.detect([image], verbose=0)[0]
                print('detect time: ', time.time() - t1)
                cv_window_name = "Mask-RCNN for PLC"
                cv2.namedWindow(cv_window_name)
                # cv2.moveWindow(cv_window_name, 10, 10)
                class_names = ['BG', 'Simatic IOT2000', 'Raspberry Pi', 'Nuc', 'DR-120-24', 'RS30']
                # print('redult:',list(r))
                mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
                                              class_names, r['scores'],score_threshold=0.2)

                cv2.imshow(cv_window_name, mask_img)
                cv2.waitKey(1)
                count += 1
            # while (True):
            #     try:
            #         image = camera.read()
            #         image = preprocess_image(image, 720, 1280)
            #         t1 = time.time()
            #         r = model.detect([image], verbose=0)[0]
            #         print('detect time: ', time.time() - t1)
            #         cv_window_name = "Mask-RCNN for PLC"
            #         cv2.namedWindow(cv_window_name)
            #         cv2.moveWindow(cv_window_name, 10, 10)
            #         class_names = ['BG', 'Simatic IOT2000', 'Raspberry Pi', 'Nuc', 'RS30', 'DR-120-24']
            #         # print('redult:',list(r))
            #         mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
            #                                       class_names, r['scores'])
            #
            #         cv2.imshow(cv_window_name, mask_img)
            #         cv2.waitKey(1)
            #         count += 1
            except:
                print("video stop!")
                break
        cv2.destroyAllWindows()

    # print("Saved to ", file_name)
############################################################
#  Training
############################################################


if __name__ == '__main__':

    # import argparse
    #
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description='Train Mask R-CNN on MS COCO.')
    # parser.add_argument("command",
    #                     metavar="<command>",
    #                     help="'train' or 'evaluate' on MS COCO")
    # parser.add_argument('--dataset', required=True,
    #                     metavar="/path/to/coco/",
    #                     help='Directory of the MS-COCO dataset')
    # parser.add_argument('--year', required=False,
    #                     default=DEFAULT_DATASET_YEAR,
    #                     metavar="<year>",
    #                     help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    # parser.add_argument('--model', required=True,
    #                     metavar="/path/to/weights.h5",
    #                     help="Path to weights .h5 file or 'coco'")
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--limit', required=False,
    #                     default=500,
    #                     metavar="<image count>",
    #                     help='Images to use for evaluation (default=500)')
    # parser.add_argument('--download', required=False,
    #                     default=False,
    #                     metavar="<True|False>",
    #                     help='Automatically download and unzip MS-COCO files (default=False)',
    #                     type=bool)
    # args = parser.parse_args()

    # print("Command: ", args.command)
    # print("Model: ", args.model)
    # print("Dataset: ", args.dataset)
    # print("Year: ", args.year)
    # print("Logs: ", args.logs)
    # print("Auto Download: ", args.download)


    # Configurations
    if args['command'] == "train":
        config = PLCConfig()
    else:
        class InferenceConfig(PLCConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            POST_NMS_ROIS_INFERENCE = 20
            # IMAGE_MIN_DIM = int(400)
            # IMAGE_MAX_DIM = int(512)
        config = InferenceConfig()
    config.display()

    # Create model
    if args['command'] == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args['logs'])
    else:

        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args['logs'])

    # Select weights file to load
    if args['model'].lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args['model'].lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args['model'].lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args['model']

    # train or test
    if args['command'] == "train":
        # Load weights


        # model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/logs/plc20180815T2316/mask_rcnn_plc_0011.h5'

        print("Loading weights ", model_path)
        # model.load_weights(model_path, by_name=True)


        # Exclude the last layers because they require a matching
        # number of classes
        model_path = '/home/jianfenghuang/Myproject/CarPlateRecog/data/mask_rcnn_plc_0038.h5'
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

        train(model)
    elif args['command'] == "test":
        #
        # test_model_path = '/home/jianfenghuang/Myproject/CarPlateRecog/data/mask_rcnn_plc_0111.h5' # recognize car plate
        test_model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/mask_rcnn_plc_0063.h5' # recognize letter
        # test_model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/mask_rcnn_plc_0353.h5'
        print("Loading weights ", test_model_path)
        model.load_weights(test_model_path, by_name=True)
        # imgPath = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/colorimage_1.jpg'
        # image_dir = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/json_train2'

        # image_dir = '/home/jianfenghuang/Pictures/picture'
        # image_dir = '/home/jianfenghuang/Pictures/numbers'
        # image_dir = '/home/jianfenghuang/Myproject/CarPlateRecog/data/ECB/ECB02_195'
        image_dir = '/home/jianfenghuang/Pictures/hongkong'
        image_dir = '/home/jianfenghuang/Pictures/3072'
        # model.keras_model.outputs=
        detect(model, image_dir=image_dir,Min_score = 0.27,watikey=True)
        print('detecting...')
        # detect(model,video_path='/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/VID_20180730_095913.mp4')


        # # from camera
        # vs = WebcamVideoStream(src=1).start()
        # detect(model, camera=vs)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
