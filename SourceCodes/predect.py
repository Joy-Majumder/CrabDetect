from ultralytics import YOLO

model = YOLO("/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/CustomModel/Yolo26Custom.pt")
model.predict(source="/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/CamImage/2.png", show=True, save=True,conf=0.6, project="/Results/",name="prediction", save_crop=True)