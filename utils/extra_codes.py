@staticmethod
def _get_image_list_mscoco(dbdir, imtype):
    """
        dbdir: Input Dataset Directory for MS COCO Dataset
        imtype: Input image type jpg or png 
    """
    train_db = dict()
    valid_db = dict()

    train_impath = os.path.join(dbdir, "train2014")
    valid_impath = os.path.join(dbdir, "val2014")
    train_lbpath = os.path.join(dbdir, "annotations", "instances_train2014.json")
    valid_lbpath = os.path.join(dbdir, "annotations", "instances_val2014.json")

    train_imlist = glob(os.path.join(train_impath, "*.{}".format(imtype.lower())))
    valid_imlist = glob(os.path.join(valid_impath, "*.{}".format(imtype.lower())))

    print("Found {} Images for Training".format(len(train_imlist)))
    print("Found {} Images for Validation".format(len(valid_imlist)))

    # Make Training Database
    with open(train_lbpath, "r") as jptr:
        anno_obj = json.load(jptr)
        for iobj in anno_obj["images"]:
            print(iobj)


    # for ipath in train_imlist:
    #     basename = os.path.basename(ipath)
    #     print(basename)
        

    # Make Validation Database
    

    return train_db, valid_db

