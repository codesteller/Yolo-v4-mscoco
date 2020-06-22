from utils.ipp import IPP


train_records = "/home/codesteller/datasets/mscoco/coco_dataset_records/train2014.record"
valid_records = "/home/codesteller/datasets/mscoco/coco_dataset_records/val2014.record"

input_db = IPP(train_records=train_records, valid_records=valid_records)

train_dataset, valid_dataset = input_db.load_records()

print(train_dataset)
print(valid_dataset)
