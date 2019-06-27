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

# change the mode
command='test'
# command='train'
img_size=[540,960]
img_size=[720,1280]
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
    NUM_CLASSES = 1 + 5  # COCO has 80 classes

    STEPS_PER_EPOCH = 1

    # base net, resnet101 or resnet50
    BACKBONE = 'resnet50'
    LEARNING_RATE = 0.0001

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
        self.add_class("PLC", 2, "B")
        self.add_class("PLC", 3, "C")
        self.add_class("PLC", 4, "D")
        self.add_class("PLC", 5, "E")


        # Train or validation dataset?
        assert subset in ["json_train", "json_val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        count=0
        for a in annotations:
            count+=1
            # if count>100:
            #     print('skip!')
            #     break
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # print('regions：',a['regions'].values)
            polygons = [a['regions'][r]['shape_attributes'] for r in range(len(a['regions']))]
            # polygons = [r['shape_attributes'] for r in a['regions'].values()]
            names = [a['regions'][r]['region_attributes']['name'] for r in range(len(a['regions']))]
            # 序列字典
            name_dict = {"A": 1, "B": 2, "C": 3,"D":4,"E":5}
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
        self.add_class("PLC", 1, "A")
        self.add_class("PLC", 2, "B")
        self.add_class("PLC", 3, "C")
        self.add_class("PLC", 4, "D")
        self.add_class("PLC", 5, "E")


        # Train or validation dataset?
        assert subset in ["json_train", "json_val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        inputjson_f = open(os.path.join(dataset_dir, subset), 'r')
        line = inputjson_f.readline()
        while line != "":
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
                names.append(obj_name)
                obj_pts = int(lines_items[obj_start_index + 1])
                polygon['all_points_x'] = [int(lines_items[obj_start_index + 1 + j]) for j in range(1, obj_pts + 1)]
                polygon['all_points_y'] = [int(lines_items[obj_start_index + 1 + j]) for j in
                                         range(obj_pts + 1, obj_pts * 2 + 1)]
                polygons.append(polygon)
                print('img:', img_name, 'obj:', obj_name, 'obj pts:',
                      [int(lines_items[obj_start_index + 1 + j]) for j in range(1, obj_pts * 2 + 1)])
                obj_start_index += obj_pts * 2 + 2
            line = inputjson_f.readline()

            name_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
            name_id = [name_dict[n] for n in names]


            image_path = os.path.join(dataset_dir, img_name)
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
    dataset_train.load_PLC(args['dataset'], "json_train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PLCDataset()
    dataset_val.load_PLC(args['dataset'], "json_val")
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

def display_mask_image(cv_window_name,image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,min_score=0.8):
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
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

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
    scoreMin=min_score
    masked_image=image.astype(np.uint8).copy()
    for i in range(N):
        if scores[i] > scoreMin:
            # Mask
            mask = masks[:, :, i]
            # color = colors[i]
            color = colors[class_ids[i]]
            # color =
            if show_mask:
                masked_image = visualize.apply_mask(image, mask, color,alpha=0.4)

    masked_image = masked_image.astype(np.uint8).copy()
    for i in range(N):
        if scores[i]>scoreMin:
            # color = colors[i]
            color = colors[class_ids[i]]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                image = cv2.rectangle(masked_image, (x1, y1),(x2, y2),(100,20,100),thickness=1)
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
                caption = "{}  {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]

            # image = cv2.addText(image,caption, (x1+10, y1+10),'Times',pointSize=10,color=color)
            # ax.text(x1, y1 + 8, caption,
            #         color='w', size=11, backgroundcolor="none")


            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = visualize.find_contours(padded_mask, 0.5)
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


def detect(model, image_path=None, video_path=None,image_dir = None,camera=None,min_score=0.5,write_img=False,waitkey=True):
    assert image_path or video_path or image_dir or camera
    if image_dir:
        imgs = os.listdir(image_dir)
        write_img=write_img
        for img in imgs:
            if img[-4:]!=".jpg" and img[-4:]!=".png" and img[0]!=".":
                continue
            print("image:",img)
            # Run model detection and generate the color splash effect
            print("Running on {}".format(img))
            # Read image
            try:
                image = skimage.io.imread(image_dir+'/'+img)
            except:
                print('image error:',img)
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            t1 = time.time()
            r = model.detect([image], verbose=1)[0]
            print('detect time: ', time.time() - t1)
            cv_window_name = "Mask-RCNN for PLC"
            # cv2.namedWindow(cv_window_name)
            # cv2.moveWindow(cv_window_name, 10, 10)
            class_names = ['BG', 'Simatic IOT2000', 'Raspberry Pi', 'Nuc', 'DR-120-24', 'RS30']
            # print('redult:',list(r))
            mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
                                          class_names, r['scores'], show_bbox=False)
            # print('PLC_data/masked_p1/masked_'+img[:-4])
            if write_img:
                if(len(r['class_ids'])>0):
                    cv2.imwrite('/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/plc_images_masked/'+img[:-4]+'.png',mask_img)
            # foutput.writelines(img + ':\n')
            str = r['rois']
            np.savetxt('PLC_data/masked_txt/'+img[:-4]+'.txt',str, fmt='%.2f')
            arr = np.zeros(img_size)

            cv2.imshow(cv_window_name, mask_img)
            if waitkey:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)

            # for index in range(r['masks'].shape[2]):
            #     if r['scores'][index]>0.8:
            #         mask = r['masks'][:, :, index].reshape(img_size)
            #         arr[mask] = 255
            # pic = visualize.m_apply_mask(image,r['masks'])



            # if write_img:
            #     cv2.imwrite('PLC_data/bin_map/'+img[:-4]+'.jpg',arr)



            # np.savetxt('test.txt', a, fmt='%.2f')
            # foutput.writelines('\n')

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
        mask_img = display_mask_image(cv_window_name, image, r['rois'], r['masks'], r['class_ids'],
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
                                                       class_names, r['scores'])

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
                                              class_names, r['scores'])

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
            # IMAGE_RESIZE_MODE = "none"
            IMAGE_MIN_DIM = int(400)
            IMAGE_MAX_DIM = int(512)
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

        # from keras.models import load_model
        #
        # model = load_model('/home/jianfenghuang/Documents/my_model.h5', compile=False)
        # print('load ok')
        # model.save('./my_model4.h5')
        # Load weights
        model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/logs/plc20180802T2201/mask_rcnn_plc_0029.h5'
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)
        # model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/mask_rcnn_coco.h5'
        # Exclude the last layers because they require a matching the number of classes
        # model.load_weights(model_path, by_name=True, exclude=[
        #     "mrcnn_class_logits", "mrcnn_bbox_fc",
        #     "mrcnn_bbox", "mrcnn_mask"])


        # model.keras_model.save('h5test.h5')
        # print('save done!')
        # a=model.keras_model.get_config()
        train(model)
    elif args['command'] == "test":


        # Get path to saved weights
        # Either set a specific path or find last trained weights



        test_model_path = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/logs/plc20180802T2201/mask_rcnn_plc_0150.h5'



        # export_to_pb(model, test_model_path)

        print("Loading weights ", test_model_path)
        model.load_weights(test_model_path, by_name=True)
        image_dir = '/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/color_p1'
        # image_dir = '/home/jianfenghuang/Myproject/Mask_Rcnn/mask_rcnn_C++/label_image/label_image'
        model_path='/home/jianfenghuang/Myproject/Mask_Rcnn'



        # detect(model, image_dir=image_dir,min_score=0.3,write_img=False,waitkey=False)
        print('detecting...')
        # detect(model,video_path='/home/jianfenghuang/Myproject/Mask_Rcnn/Mask_RCNN-master/PLC_data/VID_20180730_095913.mp4')


        # from camera
        vs = WebcamVideoStream(src=0).start()
        detect(model, camera=vs)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
