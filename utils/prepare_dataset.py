import os
from glob import glob
import json
import cv2
import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
import tqdm

from utils.params import DBParam
import utils.dataset_utils as dataset_util


class DB:
    def __init__(self, params):
        self.PARAMS = params
        self.PHASE_SUPPORTED = ["train2014", "val2014"]
        # None to stop vizualization, and int (n) value to do vizualization every n images
        self.viz_interval = None
        self.viz_limit = 100
        self.num_samples = None        # None to include all samples, else n samples will be taken in the dataset

    def create_records(self):
        """
            Create Tensorfloe Records file from MSCOCO, PASCAL VOC datasets
        """

        if self.PARAMS.DATASET_TYPE.lower() == "mscoco":
            if self.PARAMS.DATASET_PHASE.lower() in self.PHASE_SUPPORTED:
                train_db = self.getdb_mscoco(
                    self.PARAMS.DATASET_PATH, self.PARAMS.DATASET_PHASE)

                """
                    #TODO : Add proper records check to avoid creating multiple records
                """
                if self.PARAMS.RECORDS_CREATE:
                    self.write_records(train_db)
            else:
                print("{} dataset phase is not supported.".format(
                    self.PARAMS.DATASET_PHASE))
                exit(-1)

    def write_records(self, imdb):
        """
            Write the records file
        """
        isample = 0

        writer = tf.io.TFRecordWriter(self.PARAMS.RECORDS_PATH)
        print("Started writing records file")
        for idb in tqdm.tqdm(imdb):
            tf_example = self._create_tfdata(imdb[idb])
            writer.write(tf_example.SerializeToString())
            isample += 1
            if self.num_samples:
                if isample > self.num_samples:
                    break
        writer.close()
        self.num_samples = isample
        print("Records file written to {}".format(self.PARAMS.RECORDS_PATH))

    def write_params(self):
        """
            Write the params in a file
        """
        param_dict = dict()
        param_dict["DATASET_PATH"] = self.PARAMS.DATASET_PATH
        param_dict["DATASET_TYPE"] = self.PARAMS.DATASET_TYPE
        param_dict["DATASET_PHASE"] = self.PARAMS.DATASET_PHASE
        param_dict["RECORDS_DIR"] = self.PARAMS.RECORDS_DIR
        param_dict["IMAGE_TYPE"] = self.PARAMS.IMAGE_TYPE
        param_dict["RECORDS_CREATE"] = self.PARAMS.RECORDS_CREATE
        param_dict["IMAGE_HEIGHT"] = self.PARAMS.IMAGE_HEIGHT
        param_dict["IMAGE_WIDTH"] = self.PARAMS.IMAGE_WIDTH
        param_dict["NUM_SHARDS"] = self.PARAMS.NUM_SHARDS
        param_dict["RECORDS_PATH"] = self.PARAMS.RECORDS_PATH 
        param_dict["PARAMS_PATH"] = self.PARAMS.PARAMS_PATH 
        param_dict["NUM_SAMPLES"] = self.num_samples
        # MSCOCO Specific Flags  # ['person','dog','skateboard'] etc. None for all classes
        param_dict["search_dataset_by_category"] = self.PARAMS.search_dataset_by_category

        with open(self.PARAMS.PARAMS_PATH, "w") as fptr:
            json.dump(param_dict, fptr)
        print("PARAMS written to {} file".format(self.PARAMS.PARAMS_PATH))    

    def getdb_mscoco(self, data_dir='/mnt/raid_storage/05_Datasets/mscoco/coco_dataset', dataset_phase='train2014'):
        db_dict = dict()
        from pycocotools.coco import COCO
        annFile = '{}/annotations/instances_{}.json'.format(
            data_dir, dataset_phase)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)
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
            temp_dict["file_name"] = os.path.join(
                data_dir, dataset_phase, iobj["file_name"])
            if not os.path.exists(temp_dict["file_name"]):
                print("Image file in {} not found".format(
                    temp_dict["file_name"]))
                exit(-1)

            # Add image details like height, width, encoded_image_data
            temp_dict["height"] = iobj["height"]
            temp_dict["width"] = iobj["width"]
            temp_dict["image_format"] = os.path.splitext(
                temp_dict["file_name"])[-1].replace(".", "")

            # Add annotations to the dictionary
            # Extract Annotation Information here for every image id and pack it in the dictionary
            # in 5 lists of consisting of top left x, top left y, bottom right x, bottom right y
            # and the class labels
            x_top = list()
            x_bot = list()
            y_top = list()
            y_bot = list()
            lbl_class = list()
            num_objects = 0

            for anobj in anobjs:
                if anobj['category_id'] > 80:
                    continue
                x_top.append(anobj['bbox'][0])
                y_top.append(anobj['bbox'][1])
                x_bot.append(anobj['bbox'][2] + anobj['bbox'][0])
                y_bot.append(anobj['bbox'][3] + anobj['bbox'][1])
                lbl_class.append(anobj['category_id'])
                num_objects += 1 

            temp_dict["x_top"] = x_top
            temp_dict["y_top"] = y_top
            temp_dict["x_bot"] = x_bot
            temp_dict["y_bot"] = y_bot
            temp_dict["lbl_class"] = lbl_class
            temp_dict["num_objects"] = num_objects

            # If all categories in an image is greater than 80, then there is no 
            # segmentation label available for that image. 
            # So we do not include those images in the datatset.
            if num_objects == 0:        
                continue

            db_dict[imgId] = temp_dict

            try:
                if self.viz_interval:
                    if idx % self.viz_interval == 0:
                        print(
                            "{}. Image Id: {}  - {} objects".format(idx, imgId, len(anobjs)))
                        self._visualize_annotation(temp_dict)
                    elif idx > self.viz_limit:
                        print("Turning OFF Vizualization as there are many images")
                        self.viz_interval = False
                    else:
                        continue
            except Exception as e:
                print("Turning OFF Vizualization due to Error - {}".format(e))
                self.viz_interval = False

        return db_dict

    @staticmethod
    def _visualize_annotation(info_dict):
        image_ = cv2.imread(info_dict["image_path"])
        window_name = 'Annotated Image'
        color = (255, 0, 0)     # BGR Color for all objects
        thickness = 2

        for idx in range(info_dict["num_objects"]):
            top_left = (int(info_dict["x_top"][idx]),
                        int(info_dict["y_top"][idx]))
            bot_right = (int(info_dict["x_bot"][idx]),
                         int(info_dict["y_bot"][idx]))
            image_ = cv2.rectangle(
                image_, top_left, bot_right, color, thickness)

        # Displaying the image
        cv2.imshow(window_name, image_)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return

    def _create_tfdata(self, db_dict):
        """
            imgpath: path to the image -> only JPG/JPEG or PNG is supported
            lblpath: path to the corresponding label file, or object
            label_type: COCO dataset("mscoco") & pascal VOC 2012("voc2012") is available now
            Adapted from original repo 
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
        """

        im_info = self._parse_image_info(db_dict)

        tf_data = tf.train.Example(features=tf.train.Features(feature={
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
        return tf_data

    def _parse_image_info(self, db_dict):
        """
            Original Code from 
            https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
            TODO: Populate the following variables from PASCAL VOC 2012 Dataset.
        """

        im_info = dict()

        im_info["height"] = db_dict["height"]  # Image height
        im_info["width"] = db_dict["width"]  # Image width
        # Filename of the image. Empty if image is not from file
        im_info["filename"] = db_dict["file_name"].encode() 

        if db_dict["image_format"].lower() == "jpg" or db_dict["image_format"].lower() == "jpeg":
            im_info["image_format"] = b"jpeg"  # b'jpeg' or b'png'
        elif db_dict["image_format"].lower() == "png":
            im_info["image_format"] = b"png"

        # Encoded image bytes Load image with opencv
        im_info["encoded_image_data"] = cv2.imread(db_dict["file_name"]).tobytes()

        # List of normalized left x coordinates in bounding box (1 per box)
        im_info["xmins"] = (np.array(db_dict["x_top"])/db_dict["width"]).tolist()
        # List of normalized right x coordinates in bounding box
        im_info["xmaxs"] = (np.array(db_dict["x_bot"])/db_dict["width"]).tolist()
        # (1 per box)
        # List of normalized top y coordinates in bounding box (1 per box)
        im_info["ymins"] = (np.array(db_dict["y_top"])/db_dict["height"]).tolist()
        # List of normalized bottom y coordinates in bounding box
        im_info["ymaxs"] = (np.array(db_dict["y_bot"])/db_dict["height"]).tolist()
        # (1 per box)
        # List of string class name of bounding box (1 per box)
        im_info["classes_text"], _ = self._get_class_text_color(db_dict["lbl_class"])
        # List of integer class id of bounding box (1 per box)
        im_info["classes"] = db_dict["lbl_class"]

        return im_info

    @staticmethod
    def _get_class_text_color(idxs):
        """
        input list of class/category labels
        returns a list of corresponding class names, list of corresponding class colors 

        """
        class_list = ["unknown", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                      "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                      "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                      "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                      "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                      "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                      "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
                      "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                      "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                      "hair drier", "toothbrush"]

        class_color = [[0.0, 0.0, 0.0],
                       [0.9798433927829561, 0.4900810036808062, 0.48703574838407965],
                       [0.42705548514888647, 0.5848194916997833, 0.6847880376642885],
                       [0.7747832919122585, 0.9028527001933522, 0.5808741659323718],
                       [0.47441727233292474, 0.8615334487317498, 0.6558150237176295],
                       [0.9135895626086652, 0.5373235995485557, 0.6334472131497391],
                       [0.7465033149934714, 0.9391712392338385, 0.6046859803219653],
                       [0.946865881808655, 0.9060527026154037, 0.9117882497137265],
                       [0.4746226098249592, 0.7999080632323109, 0.7934424603571724],
                       [0.5983290248561671, 0.7587483005987041, 0.9724506965995173],
                       [0.40863733014498915, 0.5465780609739825, 0.7410962838461099],
                       [0.530557795630785, 0.6861405455216196, 0.6596975356613083],
                       [0.6032032833681259, 0.9261617858346025, 0.751705850109707],
                       [0.543274991749409, 0.5882375593585905, 0.9193653257565298],
                       [0.4615105143640116, 0.8053341683161125, 0.9771529568471382],
                       [0.7566685317425099, 0.7289966954376553, 0.6751546169484839],
                       [0.622998489498124, 0.6468864044693553, 0.7414894092381579],
                       [0.7055917515747441, 0.5574831711002401, 0.4180398318583947],
                       [0.5050009489551607, 0.6130052818080439, 0.4749446416279043],
                       [0.6243327890865633, 0.642660502099752, 0.4630944337566228],
                       [0.6813641324234738, 0.5210967715573656, 0.7306205804748178],
                       [0.48083545497601615, 0.7423509439205315, 0.5151835307531728],
                       [0.5527683771045049, 0.4992723682184795, 0.8471667604942885],
                       [0.7858563450339742, 0.9984948575272664, 0.8432045166855131],
                       [0.7032003172405629, 0.6201665492007924, 0.4229577165169052],
                       [0.6438081828028359, 0.8449373646454723, 0.4678950605850313],
                       [0.45330194429864185, 0.760668592367858, 0.7387858458946805],
                       [0.6014653018324823, 0.9827858973700554, 0.9942666861997519],
                       [0.5227000841070419, 0.7149674406851377, 0.6265663932232851],
                       [0.8536688431772681, 0.9314039631806175, 0.7062008853066015],
                       [0.8970356376769368, 0.7088446511165533, 0.9298501567960383],
                       [0.43477687729966497, 0.6797056019083867, 0.6978567147822508],
                       [0.9758156376995148, 0.6341254168196968, 0.7056555144363447],
                       [0.5722606117630404, 0.48949041941400423, 0.9874770374015426],
                       [0.5522156971442957, 0.8599904263886635, 0.9303554370441993],
                       [0.5999902904059907, 0.9857529453370026, 0.6563129741468261],
                       [0.4562280795004887, 0.7180602651065435, 0.6974413764836176],
                       [0.4493180888896192, 0.8333866753738608, 0.8121911525214123],
                       [0.757643489919546, 0.5762510499445238, 0.9967159262946843],
                       [0.5202335545991799, 0.904386840733893, 0.6087318813552611],
                       [0.8818234637712765, 0.6145264773059099, 0.5100927232112377],
                       [0.5502275184601111, 0.9663171086051565, 0.8591753722509388],
                       [0.8208707413896483, 0.7361175820906594, 0.5863722187713001],
                       [0.5994815638801585, 0.6648663209073793, 0.9779452764814531],
                       [0.7298156728709512, 0.511536940665275, 0.8465196289714878],
                       [0.9789582418156337, 0.9720904369081024, 0.49033162956786586],
                       [0.7193598129256522, 0.8389894607496223, 0.5430409314889733],
                       [0.8725543882217767, 0.9954174871396996, 0.965413844175224],
                       [0.6131519076540886, 0.5684481415206403, 0.7571573346763036],
                       [0.8772065893446338, 0.6676103406866454, 0.8107811474506861],
                       [0.45873477505103766, 0.8992988643625196, 0.7926030719858598],
                       [0.4198863878557294, 0.7538972113097349, 0.798738584961623],
                       [0.5419501608992475, 0.8930717309452473, 0.8236605566898123],
                       [0.8565253979890743, 0.6659089308604046, 0.9510047483009797],
                       [0.841375370745508, 0.9674999424722052, 0.7390439583583867],
                       [0.810777227204933, 0.5060626626751077, 0.8355543505280153],
                       [0.7817669883230178, 0.49282967554550705, 0.7421693536774052],
                       [0.9447550947007866, 0.9255983463396024, 0.53696312939806],
                       [0.7821907762521996, 0.5382642164883905, 0.4863054602786372],
                       [0.48582375422152013, 0.5681864184241766, 0.5230762446416527],
                       [0.5474797894491359, 0.6186071301986436, 0.5465184406455933],
                       [0.602211869068892, 0.5087239217436831, 0.5610667885563263],
                       [0.6207543046615845, 0.42556335151564134, 0.7091970800115421],
                       [0.9511809794998318, 0.7790696284593395, 0.8269278523878171],
                       [0.5695529836584258, 0.7646414777612167, 0.5915916306281673],
                       [0.5336955443701689, 0.9333972943899621, 0.7144395808813309],
                       [0.7171818853349963, 0.864753985200019, 0.40496546279836054],
                       [0.6954823843009611, 0.44422709175057873, 0.8472944284949241],
                       [0.5154534033440141, 0.9695506844131129, 0.7760185630152855],
                       [0.5762106566917125, 0.8137910590680788, 0.4703726859793429],
                       [0.7195059077483638, 0.4568061819576644, 0.4565895539675168],
                       [0.8156908783181278, 0.4566273841862254, 0.9531679083157373],
                       [0.8517765567670827, 0.5791549408368833, 0.7282154176193887],
                       [0.6881688923568582, 0.8256401942446301, 0.8456669121049436],
                       [0.8080666960940575, 0.9969414836100342, 0.6649455314155894],
                       [0.9267811109880614, 0.9341489635924403, 0.9874932151401726],
                       [0.7500528287768342, 0.9959966834588785, 0.5199228816998431],
                       [0.9905965959296418, 0.42855538960784, 0.5501634074551857],
                       [0.9175653038718657, 0.4444199692271937, 0.417120936139466],
                       [0.4473009977519588, 0.6425537792889552, 0.5417517723533591],
                       [0.4011160374007472, 0.7517094895146288, 0.49843073404331145]]

        ret_list_text = list()
        ret_list_color = list()
        for idx in idxs:
            if idx <=80:
                ret_list_text.append(class_list[idx].encode())
                ret_list_color.append(class_color[idx])
            else:
                print("{} - class text is unknown".format(idx))
                ret_list_text.append(class_list[0].encode())
                ret_list_color.append(class_color[0])

        return ret_list_text, ret_list_color
