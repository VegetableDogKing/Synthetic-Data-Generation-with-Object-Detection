from ultralytics import YOLOv10

# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10/yolov10x.pt')

model.train(data='yolov10/data.yaml', epochs=500, batch=1, imgsz=960)