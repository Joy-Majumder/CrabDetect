from ultralytics import YOLO

model = YOLO("/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/model/yolo26x.pt")
model.train(data = "/Users/joy0x1/Downloads/UIU Mariner/WORK/DataSet/dataset/Custom_dataset.yaml", imgsz=640,batch= 9, epochs = 100, workers = 0, device= 0 ) 