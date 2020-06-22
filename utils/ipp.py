import tensorflow as tf
import utils.dataset_utils as dataset_util


class IPP:
    def __init__(self, train_records, valid_records):
        self.RECORDS_PATH = [train_records, valid_records]

    def load_records(self):
        train_dataset = tf.data.TFRecordDataset(self.RECORDS_PATH[0])
        valid_dataset = tf.data.TFRecordDataset(self.RECORDS_PATH[1])

        print(train_dataset)
        input_features = tf.train.Example()

        for raw_features in train_dataset.take(1):
            input_features.ParseFromString(raw_features.numpy())
            self.decode_tfdata(input_features.features)

        return train_dataset, valid_dataset

    @staticmethod
    def decode_tfdata(_features):
        im_info = dict()
        im_info["height"] = _features.feature['image/height']

        print(im_info["height"])


