import os
from glob import glob
import json
import cv2
import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt

from utils.params import DBParam
import utils.dataset_utils as dataset_util


class DB:
    def __init__(self, params):
        self.PARAMS = params
        self.PHASE_SUPPORTED = ["train2014", "val2014"]
        # None to stop vizualization, and int (n) value to do vizualization every n images
        self.viz_interval = 10
        self.viz_limit = 100      
        
    def create_records(self):

        if self.PARAMS.DATASET_TYPE.lower() == "mscoco":
            if self.PARAMS.DATASET_PHASE.lower() in self.PHASE_SUPPORTED:
                train_db = self.getdb_mscoco(self.PARAMS.DATASET_PATH, self.PARAMS.DATASET_PHASE)
            else:
                print("{} dataset phase is not supported.".format(self.PARAMS.DATASET_PHASE))
                exit(-1)

    def getdb_mscoco(self, data_dir='/mnt/raid_storage/05_Datasets/mscoco/coco_dataset', dataset_phase='train2014'):
        db_dict = dict()
        from pycocotools.coco import COCO
        annFile='{}/annotations/instances_{}.json'.format(data_dir,dataset_phase)

        # initialize COCO api for instance annotations
        coco=COCO(annFile)
        # get all Image IDs present in the annotation
        if self.PARAMS.search_dataset_by_category:
            catIds = coco.getCatIds(self.PARAMS.search_dataset_by_category)
            imgIds = coco.getImgIds(catIds=catIds)
        else:
            imgIds = coco.getImgIds()
        print(len(imgIds))

        for idx, imgId in enumerate(imgIds):
            annIds = coco.getAnnIds(imgIds=imgId)
            anobjs = coco.loadAnns(annIds)
            iobj = coco.loadImgs(imgId)[0]
            temp_dict = dict()

            # Add image path after checking if exists
            temp_dict["image_path"] = os.path.join(data_dir, dataset_phase, iobj["file_name"])
            if not os.path.exists(temp_dict["image_path"]):
                print("Image file in {} not found".format(temp_dict["image_path"]))
                exit(-1)

            # Add image details like height, width, encoded_image_data
            temp_dict["height"] = iobj["height"]
            temp_dict["width"] = iobj["width"]
            _, temp_dict["encoded_image_data"] = os.path.splitext(temp_dict["image_path"])


            # Add annotations to the dictionary
            # Extract Annotation Information here for every image id and pack it in the dictionary
            # in 5 lists of consisting of top left x, top left y, bottom right x, bottom right y
            # and the class labels
            x_top = list()
            x_bot = list()
            y_top = list()
            y_bot = list()
            lbl_class = list()

            for anobj in anobjs:
                x_top.append(anobj['bbox'][0])
                y_top.append(anobj['bbox'][1])
                x_bot.append(anobj['bbox'][2] + anobj['bbox'][0])
                y_bot.append(anobj['bbox'][3] + anobj['bbox'][1])
                lbl_class.append(anobj['category_id'])
                
            temp_dict["x_top"] = x_top
            temp_dict["y_top"] = y_top
            temp_dict["x_bot"] = x_bot
            temp_dict["y_bot"] = y_bot
            temp_dict["lbl_class"] = lbl_class
            temp_dict["num_objects"] = len(anobjs)

            db_dict[imgId] = temp_dict

            if self.viz_interval:
                if idx % self.viz_interval == 0:
                    print("{}. Image Id: {}  - {} objects".format(idx, imgId, len(anobjs)))
                    self._visualize_annotation(temp_dict)
                elif idx > self.viz_limit:
                    print("Turning OFF Vizualization as there are many images")
                    self.viz_interval = False
                else:
                    continue

        return db_dict 

    @staticmethod
    def _visualize_annotation(info_dict):
        image_ = cv2.imread(info_dict["image_path"]) 
        window_name = 'Annotated Image'
        color = (255, 0, 0)     # BGR Color for all objects
        thickness = 2

        for idx in range(info_dict["num_objects"]):
            top_left = (int(info_dict["x_top"][idx]), int(info_dict["y_top"][idx]))
            bot_right = (int(info_dict["x_bot"][idx]), int(info_dict["y_bot"][idx]))
            image_ = cv2.rectangle(image_, top_left, bot_right, color, thickness)
        
        # Displaying the image  
        cv2.imshow(window_name, image_) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return


    def _create_tfdata(self, imgpath, lblpath, label_type="mscoco"):
        """
            imgpath: path to the image -> only JPG/JPEG or PNG is supported
            lblpath: path to the corresponding label file, or object
            label_type: COCO dataset("mscoco") & pascal VOC 2012("voc2012") is available now
            Adapted from original repo 
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
        """
        
        if label_type.lower() == "mscoco":
            im_info = self._parse_mscoco(imgpath, lblpath)
        elif label_type.lower() == "voc2012":
            im_info = self._parse_voc2012(imgpath, lblpath)
        else:
            print("Label type {} is not supported. Only mscoco or voc2012 is supported".format(label_type))
            exit(-1)


        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(im_info["height"]),
            'image/width': dataset_util.int64_feature(im_info["width"]),
            'image/filename': dataset_util.bytes_feature(im_info["filename"]),
            'image/source_id': dataset_util.bytes_feature(im_info["filename"]),
            'image/encoded': dataset_util.bytes_feature(im_info["encoded_image_data"]),
            'image/format': dataset_util.bytes_feature(im_info["image_format"]),
            'image/object/bbox/xmin': dataset_util.float_list_feature(im_info["xmins"]),
            'image/object/bbox/xmax': dataset_util.float_list_feature(im_info["xmaxs"]),
            'image/object/bbox/ymin': dataset_util.float_list_feature(im_info["ymins"]),
            'image/object/bbox/ymax': dataset_util.float_list_feature(im_info["ymaxs"]),
            'image/object/class/text': dataset_util.bytes_list_feature(im_info["classes_text"]),
            'image/object/class/label': dataset_util.int64_list_feature(im_info["classes"]),
        }))
        return tf_example

    @staticmethod
    def _parse_voc2012(imgpath, lblpath):
        """
            Original Code from 
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
            TODO: Populate the following variables from PASCAL VOC 2012 Dataset.
        """

        im_info = dict()
        
        im_info["height"] = None # Image height
        im_info["width"] = None # Image width
        im_info["filename"] = None # Filename of the image. Empty if image is not from file
        im_info["encoded_image_data"] = None # Encoded image bytes
        im_info["image_format"] = None # b'jpeg' or b'png'

        im_info["xmins"] = [] # List of normalized left x coordinates in bounding box (1 per box)
        im_info["xmaxs"] = [] # List of normalized right x coordinates in bounding box
                    # (1 per box)
        im_info["ymins"] = [] # List of normalized top y coordinates in bounding box (1 per box)
        im_info["ymaxs"] = [] # List of normalized bottom y coordinates in bounding box
                    # (1 per box)
        im_info["classes_text"] = [] # List of string class name of bounding box (1 per box)
        im_info["classes"] = [] # List of integer class id of bounding box (1 per box)

        return im_info

    @staticmethod
    def _parse_mscoco(imgpath, lblpath):
        """
            Original Code from 
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
            TODO: Populate the following variables from MSCOCO Dataset.
        """

        im_info = dict()
        
        im_info["height"] = None # Image height
        im_info["width"] = None # Image width
        im_info["filename"] = None # Filename of the image. Empty if image is not from file
        im_info["encoded_image_data"] = None # Encoded image bytes
        im_info["image_format"] = None # b'jpeg' or b'png'

        im_info["xmins"] = [] # List of normalized left x coordinates in bounding box (1 per box)
        im_info["xmaxs"] = [] # List of normalized right x coordinates in bounding box
                    # (1 per box)
        im_info["ymins"] = [] # List of normalized top y coordinates in bounding box (1 per box)
        im_info["ymaxs"] = [] # List of normalized bottom y coordinates in bounding box
                    # (1 per box)
        im_info["classes_text"] = [] # List of string class name of bounding box (1 per box)
        im_info["classes"] = [] # List of integer class id of bounding box (1 per box)

        return im_info
        

