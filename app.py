import numpy as np
import os
from yolo_resnet import resnet
import flask
from flask import Flask,request,jsonify,url_for,render_template
import torch
import cv2
import base64
from utils import nms
#from werkzeug import secure_filename

app=Flask(__name__,template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
UPLOAD_FOLDER="./static"
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER



#model_path=os.path.join(app.config["UPLOAD_FOLDER"],"parameters.pb")
with torch.no_grad():
    model=resnet()

      
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


def predict(inp,path):
    with torch.no_grad():
        img1=inp
        img=torch.tensor([img1],dtype=torch.float32)
        img=img.permute((0,3,1,2))
        out=model(img)
        print("predicteddddddddddddddddddddddddddd")
        print(out.shape)

        filtered=nms(out,C=2,conf_ph=0.22,iou_ph=0.5)
        for image in filtered:
            for n in range(2):
                if len(image[n]):
                    boxes=image[n][0]
                    for box in boxes:
                        
                        x1=int(box[0]*448)
                        y1=int(box[1]*448)
                        x2=int(box[2]*448)
                        y2=int(box[3]*448)
                        
                        if (x1>=0) and (y1>=0) and (x2>=0) and (y2>=0) and x1!=x2 and  y1!=y2 :
                            if (x1<448) and (x2<448) and (y1<448) and (y2<448):
                                print("hiiiiiii",box,n)
                                if n==0:
                                    img1=cv2.rectangle(img1, (x1,y1), (x2,y2), color=(0,0,255),thickness=2)
                                    img1=cv2.putText(img1,"fire",(x1,y1),color=(0,0,255),fontFace=0,fontScale=0.5,thickness=2)
                                if n==1:
                                    img1=cv2.rectangle(img1, (x1,y1), (x2,y2), color=(0,255,0),thickness=2)
                                    img1=cv2.putText(img1,"smoke",(x1,y1),color=(0,255,0),fontFace=0,fontScale=0.5,thickness=2)
            
        cv2.imwrite(path,img1)
        print("file updated")
    return



    


@app.route('/',methods=['GET','POST'])
def home():
    if flask.request.method =="GET":
        return render_template("index.html")
    else:
        f=request.files["image"]
        path=os.path.join(app.config["UPLOAD_FOLDER"],f.filename)

        f.save(path)

        input_img=cv2.imread(path)
        input_img=cv2.resize(input_img,(448,448),interpolation=cv2.INTER_CUBIC)
        inp_path=os.path.join(app.config["UPLOAD_FOLDER"],"inp.png")
        cv2.imwrite(inp_path,input_img)

        pred_path=os.path.join(app.config["UPLOAD_FOLDER"],"pred.png")
        predict(input_img,pred_path)

        pred_path="./static/pred.png"

        return render_template("upload.html",img1=inp_path,img2=pred_path)


@app.route('/mobile',methods=['GET','POST'])
def mobile():
    if flask.request.method =="GET":
        return render_template("index.html")
    else:
        print(request.files)
        f=request.files["image"]
        path=os.path.join(app.config["UPLOAD_FOLDER"],f.filename)

        f.save(path)

        input_img=cv2.imread(path)
        input_img=cv2.resize(input_img,(448,448),interpolation=cv2.INTER_CUBIC)
        inp_path=os.path.join(app.config["UPLOAD_FOLDER"],"inp.png")
        cv2.imwrite(inp_path,input_img)

        pred_path=os.path.join(app.config["UPLOAD_FOLDER"],"pred.png")
        predict(input_img,pred_path)

        pred_path="./static/pred.png"


        
        img = cv2.imread(pred_path) # reads the PIL image
        retval, buffer = cv2.imencode('.jpg', img)
        retval=None
        img_base64 = base64.b64encode(buffer)
        return img_base64

if __name__ == "__main__":
    print("app started")
    app.run(debug=False)
    
    pass
