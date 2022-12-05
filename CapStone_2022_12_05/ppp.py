# 라이브러리
import numpy as np
import cv2
import time
from PIL import Image
from edgetpu.detection.engine import DetectionEngine
import os

# 파일의 폴더 경로를 가져오기
model_dir = os.path.dirname(os.path.realpath(__file__))

# COCO 데이터 세트 에 대해 훈련 된 물체 감지 모델
model = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"

# 라벨 텍스트 ex) 0 person
label_path = "coco_labels.txt"

# 파일의 폴더 경로와 edgetpu 모델을 join 시킴
M_Dir = os.path.join(model_dir, model)
L_Dir = os.path.join(model_dir, label_path)

# DetectionEngine 생성
engine = DetectionEngine(M_Dir)

# 초기화
labels = {}
box_colors = {}
prevTime = 0

# 사람 카운터
person_flag = 0                     # 전역 변수로써 사람을 체크하는 flag를 추가함

# 라벨 파일 읽기
with open(L_Dir, 'r') as f :
    lines = f.readlines()
    for line in lines:                              # ex) '87 teddy bear'
        id, name = line.strip().split(maxsplit=1)   # ex) '87', 'teddy bear'
        labels[int(id)] = name

print("Quit to ESC.")       # 출력

cap = cv2.VideoCapture(-1)  # 카메라 열기
# vid = cv2.VideoCapture("danger.mp4")

# 카메라가 열려 있는 동안
while cap.isOpened() :

    # ret, frame 읽기 
    ret, frame = cap.read()
    
    # 카메라 정보를 읽지 못하면 종료
    if not ret : break

    # BGR을 RGB로 변환
    img = frame[:, :, ::-1].copy()
    # NumPy 배열을 PIL 이미지로 변환 
    img = Image.fromarray(img)

    # threshold = 0.5 : 최소 신뢰도, top_k = 5: 최대 감지 개체 수
    candidates = engine.detect_with_image(img, threshold = 0.5, top_k = 5, keep_aspect_ratio = True, relative_coord = False, )
    
    if candidates:

        for obj in candidates:
            
            # 같은 객체면 같은 색상으로 나오기
            if obj.label_id in box_colors : box_color = box_colors[obj.label_id]
            
            # 새로운 객체면 랜덤 색상으로 나오기
            else :
                box_color = [int(j) for j in np.random.randint(0,255, 3)]
                box_colors[obj.label_id] = box_color

            # 사각 박스 그리기
            box_left, box_top, box_right, box_bottom = tuple(map(int, obj.bounding_box.ravel()))
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, 2)

            # 라벨 박스 그리기
            accuracy = int(obj.score * 100)
            label_text = labels[obj.label_id] + " (" + str(accuracy) + "%)" 
            (txt_w, txt_h), base = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 2, 3)
            cv2.rectangle(frame, (box_left - 1, box_top - txt_h), (box_left + txt_w, box_top + txt_h), box_color, -1)
            cv2.putText(frame, label_text, (box_left, box_top + base), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

        # 사람을 인식 했을 경우 ( + 사람을 처음 인식했을 경우 )
        if labels[obj.label_id] in 'person' and person_flag == 0:
            # 사람을 인식했다는 의미로, person_flag를 1로 변경
            person_flag = 1
            print("Person detected")                # 확인용 출력문 1

            vid = cv2.VideoCapture("danger.mp4")        # While문 밖에 선언된 vid를 이곳으로 옮겨서, release되어도 사람이 인식되면 다시 vid를 불러오게 함
            # 현재 시간에서 3초
            max_t = time.time() + (3 * 1)
            
            while vid.isOpened() :
                ret1, frame1 = vid.read()

                cv2.imshow("DANGER", frame1)

                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                # 현재시간 + 3 초가 되면 비디오 객체를 해제 및 창 닫기
                if time.time() > max_t :
                    vid.release()
                    cv2.destroyWindow("DANGER")
                    break
        # 확인용 출력 2, 만약 사람이 아닌 다른 물체가 인식된다면 다른 물체도 인식된다는 메세지를 출력.
        else :
            print ("other object detected ")

    # 만약 화면에 사람이 사라졌고, 다른 물체또한 사라졌다면 그때, person_flag를 0으로 바꿔준다.
    if not candidates :
        person_flag = 0
        print("Not detected")               #확인용 출력 3

    # 사람이 처음 인식되었을 그 시각(위험 영상 출력 당시)에만 Person detected가 출력되고, 
    # 그 이후로는 사람이 계속 인식되도 other object detected만 출력됨. (if문 못들어가기 때문)
    # 사람이 아닌 다른 오브젝트가 있을 때도 other object detected가 출력됨
    # 오브젝트가 모두 안잡히게 된다면 그 때 비로소 person_flag가 0으로 초기화됨
    
            

    # 아직 미완성. ( 사람 1명만 인식된 상황에서 다른 물체(다른 사람포함)을 인식하게 되는 경우, label[object.label_id]가 값이 달라져서 
    # 사람이 순간적으로 인식이 안되서 person_flag가 바뀌고, 사람을 또 한순간 포착해서 다시 flag가 1로 바뀌어서 재인식됨. )

    # FPS 속도 계산         
    currTime = time.time()
    fps = 1/ (currTime -  prevTime)
    prevTime = currTime

    # FPS 출력
    cv2.putText(frame, "FPS : %.1f" % fps, (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    
    # 화면 보여주기
    cv2.imshow('CAMERA', frame)

    # ESC 누를 시, 종료
    if cv2.waitKey(1)&0xFF == 27 : break

cap.release()               # 메모리 해제
cv2.destroyAllWindows()     # 모든 창 종료