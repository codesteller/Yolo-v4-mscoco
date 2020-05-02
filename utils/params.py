import os


class DBParam:
    def __init__(self,
                 dataset_path="/home/codesteller/raid_storage/05_Datasets/mscoco/coco_dataset",
                 records_path="/home/codesteller/datasets/mscoco/coco_dataset_records",
                 dataset_type = "mscoco",
                 dataset_phase="train2014"):
        self.DATASET_PATH = dataset_path
        self.DATASET_TYPE = dataset_type
        self.DATASET_PHASE = dataset_phase
        self.RECORDS_PATH = records_path
        self.IMAGE_TYPE = "jpg"
        self.RECORDS_CREATE = True
        self.IMAGE_HEIGHT = 800
        self.IMAGE_WIDTH = 800
        self.NUM_SHARDS = 10
        self.DATABASE = os.path.join(self.RECORDS_PATH, "train_dataset.record")
        self.check_path()

        # MSCOCO Specific Flags
        # ['person','dog','skateboard'] etc. None for all classes
        self.search_dataset_by_category=['person']      

    def check_path(self):
        if not os.path.exists(self.RECORDS_PATH):
            os.makedirs(self.RECORDS_PATH)
        else:
            self.RECORDS_CREATE = False
