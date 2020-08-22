import tensorflow as tf
from utils.ipp import IPP
from models.od_models import Network
from distutils.version import LooseVersion


class TrainModel:
    def __init__(self, params):
        self._assert_version()
        self.params = params
        self.input_db = IPP(train_records=self.params.train_records,
                            valid_records=self.params.train_records)
        self.train_dataset, self.valid_dataset = self.input_db.load_records()
        self.network = self.build_model()


    def train_model(self):
        pass

    def build_model(self):
        net = Network(self.params.core_network)
        # Input shape is tuple -> hieght, width, num_channels
        input_shape = (self.params.data_param["IMAGE_HEIGHT"], self.params.data_param["IMAGE_WIDTH"], 3) 
        net.build_network(input_shape, self.params.corenet_trainable)
        return net

    @staticmethod
    def _assert_version():
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) >= LooseVersion('2.1'), 'Please use TensorFlow version 2.1 or newer'
        print('TensorFlow Version: {}'.format(tf.__version__)) 
        return
