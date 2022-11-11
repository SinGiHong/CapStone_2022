import numpy as np
import cv2
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine
import sys, os

model_dir = os.path.dirname(os.path.realpath(__file__))    # 이 파일의 폴더 경로를 가져옴
model = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
label_path = "coco_labels.txt"
# creating DetectionEngine with model

M_Dir = os.path.join(model_dir, model)
L_Dir = os.path.join(model_dir, label_path)
engine = DetectionEngine(M_Dir)  
    # 이 파일의 폴더 경로와 edgetpu 모델을 join 시킴

labels = {}
box_colors = {}
prevTime = 0

# reading label file
with open(L_Dir, 'r') as f:
    lines = f.readlines()
    for line in lines:    # ex) '87 teddy bear'
        id, name = line.strip().split(maxsplit=1)   # ex) '87', 'teddy bear'
        labels[int(id)] = name
print(f"Model loaded({model}) \nTrained object({len(labels)}):\n{labels.values()}")
print("Quit to ESC.")

cap = cv2.VideoCapture(-1)
while True:
    ret, frame = cap.read()
    if not ret:
        print("cannot read frame.")
        break
    img = frame[:, :, ::-1].copy()  # BGR to RGB
    img = Image.fromarray(img)  # NumPy ndarray to PIL.Image

    # threshold=0.5: mininum confidence , top_k=5 : maximum number of detected object 
    candidates = engine.detect_with_image(img, threshold=0.5, top_k=5, keep_aspect_ratio=True, relative_coord=False, )
    if candidates:
        for obj in candidates:
            if obj.label_id in box_colors:
                box_color = box_colors[obj.label_id] # the same color for the same object
            else :
                box_color = [int(j) for j in np.random.randint(0,255, 3)] # random color for new object
                box_colors[obj.label_id] = box_color

            # drawing bounding-box
            box_left, box_top, box_right, box_bottom = tuple(map(int, obj.bounding_box.ravel()))
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, 2)

            # drawing label box
            accuracy = int(obj.score * 100)
            label_text = labels[obj.label_id] + " (" + str(accuracy) + "%)" 
            (txt_w, txt_h), base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 2, 3)
            cv2.rectangle(frame, (box_left - 1, box_top - txt_h), (box_left + txt_w, box_top + txt_h), box_color, -1)
            cv2.putText(frame, label_text, (box_left, box_top+base), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    # calculating and drawing fps            
    currTime = time.time()
    fps = 1/ (currTime -  prevTime)
    prevTime = currTime
    cv2.putText(frame, "fps:%.1f"%fps, (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    cv2.imshow('Object Detecting', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break  
cap.release()