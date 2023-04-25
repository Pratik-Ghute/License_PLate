# -*- coding: utf-8 -*-

# 1. Library imports
import uvicorn
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from fastapi.responses import TemplateResponse
from typing import List, Union
from fastapi.responses import FileResponse

import shutil
import numpy as np
import cv2
import torch
import time
import re
import easyocr

from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from PIL import Image as PILImage

from typing import Annotated

from fastapi import FastAPI, File, UploadFile

# specify the device for model execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load the OCR reader
EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2
# 2. Create the app object
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
# Mount the static directory at /static
# app.mount("/static/img_op", StaticFiles(directory="/static/img_op"), name="/static/img_op")
@app.get("/")
def read(request:Request):
    return templates.TemplateResponse("index.html", {"request": request,"Hello" : "World"})


@app.get("/")
def read(request:Request):
    return templates.TemplateResponse("index.html", {"request": request,"Hello" : "World"})

@app.post("/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename
    with open(f"static/{filename}", "wb") as f:
        f.write(contents)
    file_path = f"static/{filename}"
    result, n = main(img_path = file_path)  # pass file_path to the function
    return templates.TemplateResponse("index.html", {"request": request, "file_name": filename, "result": result, "n":n})

@app.post("/uploadvideo/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename
    with open(f"static_vid/{filename}", "wb") as f:
        f.write(contents)
    file_path = f"static_vid/{filename}"
    result2, n2,total_frames, fps = main(vid_path = file_path)  #  pass file_path to the function
    return templates.TemplateResponse("index.html", {"request": request, "file_name2": filename, "result2": result2, "n2":n2, "total_frames": total_frames, "fps":fps})
# 



@app.post("/accessvideo/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    filename = Webcam_live
    with open(f"static_vid/{filename}", "wb") as f:
        f.write(contents)
    file_path = f"static_vid/{filename}"
    result3, n3 = main(vid_path = 0)  # pass file_path to the function
    return templates.TemplateResponse("index.html", {"request": request, "file_name3": filename, "result3": result3, "n3":n3})

### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]

    results = model(frame)
    

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    
    plates = []
    
    
    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
           
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            
            

            
            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
            plates += [plate_num]

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
        
            
        
    return frame, plates



#### ---------------------------- function to recognize license plate --------------------------------------


# function to recognize license plate numbers using EasyOCR
def recognize_plate_easyocr(img, coords,reader,region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
    



    ocr_result = reader.readtext(nplate)



    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

    if len(text) ==1:
        text = text[0].upper()
    return text


### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate





### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None):

    # print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5', 'custom', path="./yolov5/best.pt", source ='local') ### The repo is stored locally

    classes = model.names ### class names in string format




    
    ### --------------- for detection on image --------------------
    if img_path != None:
        # print(f"[INFO] Working with image: {img_path}")
        img_out_name = img_path

        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        frame, plates = plot_boxes(results, frame,classes = classes)
        img_out_name = f"{img_path.split('/')[-1]}"
        cv2.imwrite(f"static/{img_out_name}",frame)    

        return plates, len(plates)
      
    # Process video
    elif vid_path !=None:
        # print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)

        vid_out_name = vid_path
        vid_out_name = f"{vid_out_name.split('/')[-1]}"
        vid_out = f"static_vid/{vid_out_name}"
        ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
        out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        

        # assert cap.isOpened()
       
        f_n = 1
        # cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            
            ret, frame = cap.read()
        

            if not ret:
                break
            # print(f"[INFO] Working with frame {frame_no} ")

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = detectx(frame, model = model)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)


            frame, plates = plot_boxes(results, frame,classes = classes)
            
            # cv2.imshow("vid_out", frame)
            
                # print(f"[INFO] Saving output video. . . ")
            out.write(frame)
            f_n= f_n + 1


            # if frame_no == total_frames:
            #     break
           

        return plates, len(plates), total_frames, fps
        # print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
    out.release()
    cap.release()
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)


