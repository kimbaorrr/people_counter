# Import các thư viện cần thiết
import cv2 as cv
import pandas as pd
import numpy as np
from ultralytics import YOLO
import argparse
import os

parser = argparse.ArgumentParser()

# Truyền đường dẫn video
parser.add_argument("-i", "--input", help = "Video file", required=True, type=str)
parser.add_argument('-l', '--loop', help='Video loop ?', type=int)
args = parser.parse_args()

# Check file đầu vào có tồn tại hay không ?
if not os.path.exists(args.input):
    raise FileExistsError("Video không tồn tại. Vui lòng chọn file khác.!")

# Import class Tracker
from tracker import *

# Cài đặt model yolov8 đã train sẵn
model = YOLO("yolov8n.pt")

# Lưu trữ các tọa độ trỏ chuột
mouse_points = []
# Lưu trữ các vùng giới hạn
areas = []  
# Lưu trữ khung hình hiện tại (không bị mất vùng giới hạn khi draw lên frame)
current_frame = None
# Tạo màu cho vùng giới hạn [Vùng 1: Xanh dương, Vùng 2: Đỏ]
area_colors = [(255, 0, 0), (0, 0, 255)]

def draw_polygon(event, x, y, flags, param):
    """
    Vẽ tứ giác vùng giới hạn bằng tọa độ 4 điểm click chuột
    """
    global ix, iy, drawing, points, areas
    if event == cv.EVENT_LBUTTONDOWN:
        # Nếu chưa click đủ tọa độ 4 điểm để hình thành 1 tứ giác thì tiếp tục
        if len(mouse_points) != 4:
            # Thêm tọa độ của cú click chuột
            mouse_points.append((x, y))
        # Nếu đã có đủ tọa độ 4 điểm nhưng chưa đủ 2 vùng giới hạn thì tiếp tục
        if len(mouse_points) == 4 and len(areas) != 2:
            # Thêm vùng giới hạn mới
            areas.append(list(mouse_points))
            # Xóa tọa độ 4 điểm được lưu tạm thời
            mouse_points.clear()

cap = cv.VideoCapture(args.input)

cv.namedWindow("YOLO")
cv.setMouseCallback("YOLO", draw_polygon)

# Đọc file chứa các class đã phân loại theo từng dòng
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")
# print(class_list)

count = 0

# Gọi class Tracker
tracker = Tracker()

# Tạo các dictionary và set cần thiết cho việc đếm người vào và ra
people_entering = {}
entering = set()

people_exiting = {}
exiting = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # Tạo vòng lặp cho video
        if args.loop == 1:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
        #break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv.resize(frame, (1000, 500))
    current_frame = frame.copy()
    # Vẽ các vùng giới hạn lên frame & set màu cho từng vùng
    for idx, area in enumerate(areas):
        # Vẽ tứ giác vùng giới hạn [Vùng 1: Xanh dương, Vùng 2: đỏ]
        cv.polylines(current_frame, [np.array(area)], True, area_colors[idx], 3)
        # In chỉ số của vùng giới hạn
        cv.putText(
            current_frame,
            str(idx + 1),
            (area[0][0] - 5, area[0][1] - 5),
            cv.FONT_HERSHEY_DUPLEX,
            (.5),
            (0, 0, 0),
            1,
        )
    #   frame=cv.flip(frame,1)
    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype(np.int16)
    #   print(px)
    # Tạo list chứa tọa độ của bouding box
    px_list = []

    for _, row in px.iterrows():
        # print(row)
        x1 = row[0]
        y1 = row[1]
        x2 = row[2]
        y2 = row[3]
        d = row[5]
        c = class_list[d]
        if "person" in c:
            px_list.append([x1, y1, x2, y2])
    # Nếu đã đủ tọa độ của 2 vùng giới hạn thì bắt đầu tracking
    if len(areas) == 2:
        area1 = np.array(areas[0], dtype=np.int32)
        area2 = np.array(areas[1], dtype=np.int32)
        # sử dụng hàm update trong class tracker để detect và đặt id cho object trên từng frame và trả về tọa độ mới
        bbox_id = tracker.update(px_list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            # Đếm người vào
            results = cv.pointPolygonTest(
                area2, (x4, y4), False
            )  # Nếu người không đi vào vùng phát hiện 2 thì results = -1 và ngược lại
            # Nếu người đi vào vùng phát hiện 2 thì add id và tọa độ object vào dictionary số người vào
            if results >= 0:
                people_entering[id] = (x4, y4)
                cv.rectangle(
                    current_frame, (x3, y3), (x4, y4), (0, 0, 255), 2
                )  # Vẽ bouding boxs
            # Nếu người tiếp tục đi vào vùng phát hiện 1 thì đếm số người vào +1
            if id in people_entering:
                results1 = cv.pointPolygonTest(area1, (x4, y4), False)
                if results1 >= 0:
                    cv.rectangle(
                        current_frame, (x3, y3), (x4, y4), (0, 255, 0), 2
                    )  # Vẽ bouding box
                    cv.circle(
                        current_frame, (x4, y4), 5, (255, 0, 255), -1
                    )  # Vẽ chấm phát hiện
                    cv.putText(
                        current_frame,
                        f'ID: {id}',
                        (x3, y3 - 5),
                        cv.FONT_HERSHEY_DUPLEX,
                        (.5),
                        (0, 255, 255),
                        1,
                    )  # Hiện id của đối tượng
                    entering.add(id)  # Add id vào set số người vào

            # Đếm người ra
            results2 = cv.pointPolygonTest(
                area1, (x4, y4), False
            )  # Nếu người không đi vào vùng phát hiện 1 thì results = -1 và ngược lại
            # Nếu người đi vào vùng phát hiện 1 thì add id và tọa độ object vào dictionary số người ra
            if results2 >= 0:
                people_exiting[id] = (x4, y4)
                cv.rectangle(
                    current_frame, (x3, y3), (x4, y4), (0, 255, 0), 2
                )  # Vẽ bouding box
            # Nếu người tiếp tục đi vào vùng phát hiện 2 thì đếm số người ra +1
            if id in people_exiting:
                results3 = cv.pointPolygonTest(area2, (x4, y4), False)
                if results3 >= 0:
                    cv.rectangle(
                        current_frame, (x3, y3), (x4, y4), (255, 0, 255), 2
                    )  # Vẽ bouding box
                    cv.circle(
                        current_frame, (x4, y4), 5, (255, 0, 255), -1
                    )  # Vẽ chấm phát hiện
                    cv.putText(
                        current_frame,
                        f'ID: {id}',
                        (x3, y3  - 5),
                        cv.FONT_HERSHEY_DUPLEX,
                        (.5),
                        (0, 255, 255),
                        1,
                    )  # Hiện id của đối tượng
                    exiting.add(id)  # Add id vào set số người vào

        # print(people_entering)
        # print(entering)

        i = len(entering)
        o = len(exiting)

        # Hiện số người vào
        #print("People entering counted: ", i)
        cv.putText(
            current_frame,
            "Entering counted: " + str(i),
            (60, 80),
            cv.FONT_HERSHEY_DUPLEX,
            .7,
            (0, 255, 0),
            2,
        )

        # Hiện số người ra
        #print("People exiting counted: ", o)
        cv.putText(
            current_frame,
            "Exiting counted: " + str(o),
            (60, 140),
            cv.FONT_HERSHEY_DUPLEX,
            .7,
            (0, 0, 255),
            2,
        )

    cv.imshow("YOLO", current_frame)
    key = cv.waitKey(1) & 0xFF

    # Nhấn r để reset các vùng giới hạn. Vẽ lại từ đầu
    if key == ord("r"):
        mouse_points.clear()
        areas.clear()

    # Nhấn q hoặc ESC để thoát chương trình
    elif key == ord("q") or key == 27:
        break

cap.release()
cv.destroyAllWindows()
