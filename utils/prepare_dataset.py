import json
import cv2
import tensorflow as tf

from utils.params import DBParam
import utils.dataset_utils as dataset_util


class DB:
    def __init__(self, params):
        self.PARAMS = params
        
    def create_tfdata(self, imgpath, lblpath, label_type="mscoco"):
        """
            imgpath: path to the image -> only JPG/JPEG or PNG is supported
            lblpath: path to the corresponding label file, or object
            label_type: COCO dataset("mscoco") & pascal VOC 2012("voc2012") is available now
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
        

