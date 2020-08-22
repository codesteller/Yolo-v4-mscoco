import tensorflow as tf
from utils.ipp import IPP


class TrainModel:
    def __init__(self, params):
        self.params = params
        self.input_db = IPP(train_records=self.params.train_records,
                            valid_records=self.params.train_records)
        self.train_dataset, self.valid_dataset = self.input_db.load_records()

    def train_model(self):
        pass

    def build_model(self):
        pass

    
