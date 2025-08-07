from ultralytics import YOLO


val_model = YOLO('runs/detect/C2f_RepGhosts/weights/best.pt')

if __name__ == '__main__':

    val_model.val(data='ultralytics/cfg/datasets/tuna-counting.yaml')
