from flask import *
import os
import hashlib
import Database
import File
import start

app = Flask(__name__)
# app.debug = True
app.debug = False

@app.route('/',methods = ['GET','POST'])
def default():
    return redirect(url_for('show'))

# 上传功能
@app.route('/upload.php',methods = ['GET','POST'])
def upload():
    return render_template('upload.html')

# 显示目录
@app.route('/show.php',methods = ['GET'])
def show():
    return render_template('index.html',files=DB.file_list)

@app.route('/uploads.php',methods=['POST','GET'])
def uploads():
    # 读取图片
    upload_file = request.files['file']
    if not upload_file:
         return render_template('upload.html',result="upload failed!!!")
    
    upload_name = upload_file.filename.split("/")[-1]
    file = File.file(upload_name)
    upload_path = "./tmp/"+file.file_hash
    upload_file.save(upload_path)
    file.updata_path(upload_path)
    
    # 读取tag
    upload_tag = request.form['tags']
    upload_tags = upload_tag.split(" ")
    file.updata_tag(upload_tags)

    # 调用start.py放大图片
    try:
        if start.main(file.file_hash):
            file.status = "success"
    except Exception as e:
        file.status = "error"
        print(e)
    DB.add(file)
    return redirect(url_for('show'))

# 提供下载功能
@app.route("/upload/<file_hash>", methods=['GET'])
def download(file_hash):
    path = os.getcwd()+"/upload"
    print(path)
    return send_from_directory(path, file_hash+".png", as_attachment=True)
   

if __name__ == "__main__":
    DB = Database.DB()
    app.run(host='0.0.0.0',port=5000)