from utils.ipp import IPP
from utils.params import TrainParam
from utils.train import TrainModel
from absl import flags, app
import os

FLAGS = flags.FLAGS


# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('train_records', './coco_dataset_records/train2014.record',
                    'Enter the path to MSCOCO/PASCAL train records.')
flags.DEFINE_string('valid_records', './coco_dataset_records/val2014.record',
                    'Enter the path to MSCOCO/PASCAL validation records.')
flags.DEFINE_string('test_records', 'None',
                    'Enter the path to MSCOCO/PASCAL test records (optional)')


def main(argv):

    # flags.mark_flag_as_required('train_records')
    # flags.mark_flag_as_required('valid_records')

    params = TrainParam(train_records=os.path.expanduser(FLAGS.train_records),
                        valid_records=os.path.expanduser(FLAGS.valid_records),
                        test_records=os.path.expanduser(FLAGS.test_records))

    # Add core network
    # Core Network can be any one of -> "vgg19", "mobilenetv2", "resnet50"
    params.core_network = "mobilenetv2"

    train_model = TrainModel(params)
    print(train_model.train_dataset)
    print(train_model.valid_dataset)


if __name__ == "__main__":
    app.run(main)
