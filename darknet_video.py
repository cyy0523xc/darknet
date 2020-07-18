# from ctypes import *
import os
import time
import cv2
import darknet

# 模型配置
configPath = "./projects/windows-video-monitor/yolov4.cfg"
weightPath = "./projects/windows-video-monitor/backup/yolov4_final.weights"
metaPath = "./projects/windows-video-monitor/voc.data"

if not os.path.exists(configPath):
    raise ValueError("Invalid config path `" +
                     os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
    raise ValueError("Invalid weight path `" +
                     os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
    raise ValueError("Invalid data file path `" +
                     os.path.abspath(metaPath)+"`")
# batch size = 1
netMain = darknet.load_net_custom(configPath.encode("ascii"),
                                  weightPath.encode("ascii"), 0, 1)
metaMain = darknet.load_meta(metaPath.encode("ascii"))


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, wh_rate):
    for detection in detections:
        x, y, w, h = detection[2][:4]
        x, y = x*wh_rate[0], y*wh_rate[1]
        w, h = w*wh_rate[0], h*wh_rate[1]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        # msg = detection[0].decode()
        msg = detection[0].decode() + " [%.2f]" % detection[1]
        cv2.putText(img, msg, (pt1[0], pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return img


def YOLO(in_path, out_path, thresh=0.25):
    # cap = cv2.VideoCapture(0)
    print('In Video: ', in_path)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("In Video:", type(video_FourCC), video_fps, video_size)
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    input_size = (darknet.network_width(netMain),
                  darknet.network_height(netMain))
    darknet_image = darknet.make_image(input_size[0], input_size[1], 3)
    # 计算长宽缩放比
    wh_rate = (video_size[0]/input_size[0], video_size[1]/input_size[1])
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"),
                          video_fps, video_size)
    num = 0
    start = time.time()
    while True:
        num += 1
        if num % 100 == 0:
            print('Parsing %d ...' % num)

        ret, frame_read = cap.read()
        if ret is False:
            break
        if frame_read is None:
            continue
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, input_size,
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(
            netMain, metaMain, darknet_image, thresh=thresh)
        image = cvDrawBoxes(detections, frame_read, wh_rate)
        out.write(image)

    cap.release()
    out.release()
    print('Total: %d, Time: ' % num, time.time()-start)


if __name__ == "__main__":
    import fire
    fire.Fire(YOLO)
