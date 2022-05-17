from pathlib import Path as path
import torch

network_path = 'ultralytics/yolov5'
network_model = 'yolov5s'
data_path = path.cwd() / 'data' / 'heart_augmented_YOLO'

model = torch.hub.load(network_path, network_model)

test_img = data_path / 'train' / 'images' / '01G1T87PQ46Q5ZMDYNN5Y4NA89_jpeg.rf.9e113d5811d85103f3dd45f75e1fad7f.jpg'


result = model(test_img, size=416)

result.print()
result.show()