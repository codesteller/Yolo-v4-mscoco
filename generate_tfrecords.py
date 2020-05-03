from utils.params import DBParam
from utils.prepare_dataset import DB
from absl import app
from absl import flags
import os


FLAGS = flags.FLAGS


# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('dataset_path', '~/raid_storage/05_Datasets/mscoco/coco_dataset',
                    'Enter the path to MSCOCO/PASCAL VOC Dataset.')
flags.DEFINE_string('records_path', '~/datasets/mscoco/coco_dataset_records',
                    'Enter the path where to save the records.')
flags.DEFINE_string('dataset_type', 'mscoco',
                    'Enter the dataset nametype "mscoco" & "voc2012" is supported now.')
flags.DEFINE_string('dataset_phase', 'train2014',
                    'enter dataset phase train, valid or test')


def main(argv):

    # flags.mark_flag_as_required('dataset_path')
    # flags.mark_flag_as_required('records_path')

    params = DBParam(dataset_path=os.path.expanduser(FLAGS.dataset_path),
                     dataset_phase=FLAGS.dataset_phase,
                     records_path=os.path.expanduser(FLAGS.records_path),
                     dataset_type=FLAGS.dataset_type)

    _db = DB(params=params)

    # Check if Records exists. If not create records or ask user for input
    if os.path.exists(params.RECORDS_PATH):
        print("Records file already exists. Do you want to create a new records (y/N): ")
        user_input = input()
        if user_input.lower() != "y":
            print("Skipping records creation step. Record {} can be loaded".format(params.RECORDS_PATH)) 
        else:
            _db.create_records()
    else:
        _db.create_records()


if __name__ == '__main__':
    app.run(main)
