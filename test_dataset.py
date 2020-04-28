from utils.params import DBParam
from utils.prepare_dataset import DB

def test_case_1():
    _params = DBParam()
    _db = DB(_params)
    print(_db.PARAMS.DATABASE)
    _db.create_tfdata("a", "b")


if __name__ == "__main__":
    test_case_1()