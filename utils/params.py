import os
import json


class DBParam:
    def __init__(self,
                 dataset_path="/home/codesteller/raid_storage/05_Datasets/mscoco/coco_dataset",
                 records_path="/home/codesteller/datasets/mscoco/coco_dataset_records",
                 dataset_type="mscoco",
                 dataset_phase="train2014"):
        self.DATASET_PATH = dataset_path
        self.DATASET_TYPE = dataset_type
        self.DATASET_PHASE = dataset_phase
        self.RECORDS_DIR = records_path
        self.IMAGE_TYPE = "jpg"
        self.RECORDS_CREATE = True
        self.IMAGE_HEIGHT = 800
        self.IMAGE_WIDTH = 800
        self.NUM_SHARDS = 10
        self.RECORDS_PATH = os.path.join(
            self.RECORDS_DIR, self.DATASET_PHASE + ".record")
        self.PARAMS_PATH = os.path.join(
            self.RECORDS_DIR, self.DATASET_PHASE + "_params.json")
        self.check_path()

        # MSCOCO Specific Flags
        # ['person','dog','skateboard'] etc. None for all classes
        self.search_dataset_by_category = ['person']

    def check_path(self):
        if not os.path.exists(self.RECORDS_DIR):
            os.makedirs(self.RECORDS_PATH)
        """
         TODO: Add proper logic to check if records file needs to be created
        """
        # elif os.path.exists(RECORDS_PATH):
        #     self.RECORDS_CREATE = False
        return


class TrainParam:
    def __init__(self, train_records, valid_records, test_records):
        self.train_records = train_records
        self.valid_records = valid_records
        self.test_records = test_records
        # Core Network can be any one of -> "vgg19", "mobilenet", "resnet50"
        self.core_network = "resnet50"
        self.corenet_trainable = False
        self.data_param = self._load_dataparam()

    def _load_dataparam(self):
        basename = os.path.basename(self.train_records)
        dirname = os.path.dirname(self.train_records)
        basename = basename.replace(".record", "_params.json")
        param_filepath = os.path.join(dirname, basename)
        try:
            with open(param_filepath, "r") as fptr:
                param_dict = json.load(fptr)
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)
        return param_dict
