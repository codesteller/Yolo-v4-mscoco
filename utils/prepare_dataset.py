import json
import cv2
from params import DBParam


class DB:
    def __init__(self, params):
        self.PARAMS = params
        


def test_case_1():
    _params = DBParam()
    _db = DB(_params)
    print(_db.PARAMS.DATABASE)


if __name__ == "__main__":
    test_case_1()
