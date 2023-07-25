import os
import pickle
# 检查目录是否存在，不存在则创建
def get_dirs_path(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_data_pkl(data,path):
    with open(path, "w") as f:
        pickle.dump(data, f)

def load_data_pkl(path):
    with open(path, "r") as f:
        return pickle.load(f)