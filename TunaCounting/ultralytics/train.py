from ultralytics import YOLO


model = YOLO('cfg/models/Ours-nano.yaml')

if __name__ == '__main__':

    model.train(data='ultralytics/cfg/datasets/tuna-counting.yaml', epochs=300, imgsz=640,
                batch=64, optimizer='SGD', lr0=0.01, fraction=1.0)
