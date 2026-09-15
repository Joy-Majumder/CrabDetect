from ultralytics import YOLO

model = YOLO("/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/CustomModel/CustomModelVersion2.pt") # with this model v2 0.8 cant detect well 0.7 did perfectly
# model.predict(source="CamImage/1.png", show=True, save=True,conf=0.8, save_crop=True)
model.predict(source="/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/videoSource/1.mp4",show=True, conf=0.01,
    imgsz=640,
    save=True,
    verbose=True)