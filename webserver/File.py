import time
import hashlib

# 图片信息的数据结构
class file():
    def __init__(self, file_name=""):
        self.file_name = file_name
        self.time = time.strftime("%H:%M:%S %Y %b %d %a", time.localtime())
        self.file_hash = hashlib.new('md5', self.time.encode("utf-8")).hexdigest()
        self.visible = 0
        self.tag = []
        self.file_path = ""
        self.status = "NULL"

    def public(self):
        self.visible = 1

    def updata_tag(self, *tag):
        self.tag += tag

    def updata_path(self,path):
        self.file_path = path

