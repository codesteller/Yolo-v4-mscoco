import os


class DBParam:
    def __init__(self,
                 dataset_path="home/codesteller/raid_storage/05_Datasets/mscoco/coco_dataset",
                 records_path="home/codesteller/datasets/mscoco/coco_dataset_records"):
        self.DATASET_PATH = dataset_path
        self.RECORDS_PATH = records_path
        self.RECORDS_CREATE = True
        self.IMAGE_HEIGHT = 800
        self.IMAGE_WIDTH = 800
        self.NUM_SHARDS = 10
        self.DATABASE = os.path.join(self.RECORDS_PATH, "train_dataset.record")
        self.check_path()

    def check_path(self):
        if not os.path.exists(self.RECORDS_PATH):
            os.makedirs(self.RECORDS_PATH)
        else:
            self.RECORDS_CREATE = False
