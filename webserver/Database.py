import pickle
import os

class DB():
    
    def __init__(self):
        self.file_list = []
        self.save_dir = "./BK/"
        self.load()

    # 目录添加文件
    def add(self,file):
        self.file_list.append(file)
        self.save()

    # 序列化目录，存储
    def save(self):

        BK_data = pickle.dumps(self.file_list)
        if not os.path.exists(self.save_dir):
            os.mkdir("./BK/")
        with open (self.save_dir+"DB_BK","wb") as BK:
            BK.write(BK_data)

    # 读取序列化目录
    def load(self):

        if os.path.exists(self.save_dir+"/DB_BK"):
            with open(self.save_dir+"/DB_BK","rb") as BK:
                self.file_list = pickle.load(BK)
            

