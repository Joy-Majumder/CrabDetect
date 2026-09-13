from ultralytics import YOLO

model = YOLO("/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/CustomModel/CustomModel.pt")
model.predict(source="/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/CamImage/7.png", show=True, save=True,conf=0.9, save_crop=True)