import math
import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort


def normalize(frame: np.ndarray) -> np.ndarray:
    """
    Args:
        frame: BGR
    Returns: normalized 0~1 BCHW RGB
    """
    img = frame.astype(np.float32).copy() / 255.0
    img = img[:, :, ::-1]  # RGB
    img = np.transpose(img, (2, 0, 1))  # (C,H,W)
    img = np.expand_dims(img, axis=0)  # (B=1,C,H,W)
    return img


def infer_rvm_frame(weight: str = "rvm_mobilenetv3_720x1280.onnx",
                    img_path: str = "test.jpg",
                    output_path: str = "test_onnx.jpg"):
    sess = ort.InferenceSession(f'{weight}')
    print(f"Load checkpoint/{weight} done!")

    for _ in sess.get_inputs():
        print("Input: ", _)
    for _ in sess.get_outputs():
        print("Input: ", _)

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (1280, 720))  # (w, h)
    src = normalize(frame)
    r1i = np.zeros((1, 16, 135, 240), dtype=np.float32)
    r2i = np.zeros((1, 20, 68, 120), dtype=np.float32)
    r3i = np.zeros((1, 40, 34, 60), dtype=np.float32)
    r4i = np.zeros((1, 64, 17, 30), dtype=np.float32)
    bgr = np.array([0.47, 1., 0.6]).reshape((3, 1, 1))

    fgr, pha, r1o, r2o, r3o, r4o = sess.run([], {
        'src': src,
        'r1i': r1i,
        'r2i': r2i,
        'r3i': r3i,
        'r4i': r4i,
        'downsample_ratio': 0.25,
    })

    merge_frame = fgr * pha + bgr * (1. - pha)  # (1,3,H,W)
    merge_frame = merge_frame[0] * 255.  # (3,H,W)
    merge_frame = merge_frame.astype(np.uint8)  # RGB
    merge_frame = np.transpose(merge_frame, (1, 2, 0))  # (H,W,3)
    merge_frame = cv2.cvtColor(merge_frame, cv2.COLOR_BGR2RGB)

    cv2.imwrite(output_path, merge_frame)

    print(f"infer done! saved {output_path}")


def infer_rvm_video(weight: str = "rvm_mobilenetv3_720x1280.onnx",
                    video_path: str = "./demo/1917.mp4",
                    output_path: str = "./demo/1917_onnx.mp4"):
    sess = ort.InferenceSession(f'{weight}')
    print(f"Load checkpoint/{weight} done!")

    for _ in sess.get_inputs():
        print("Input: ", _)
    for _ in sess.get_outputs():
        print("Input: ", _)

    # 读取视频
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Caputer: Height: {height}, Width: {width}, Frame Count: {frame_count}")

    # 写出视频
    fps = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Create Video Writer: {output_path}")

    i = 0
    # HxW -> 720x640, 0.25 downsample
    sim_ratio = 1
    down_ratio = 0.5
    src_size = [1, 3, math.ceil(640*down_ratio), math.ceil(720*down_ratio)]
    r1_size = [1, math.ceil(16 * sim_ratio), math.ceil(src_size[2] / 2), math.ceil(src_size[3] / 2)]
    r2_size = [1, math.ceil(20 * sim_ratio), math.ceil(r1_size[2] / 2), math.ceil(r1_size[3] / 2)]
    r3_size = [1, math.ceil(40 * sim_ratio), math.ceil(r2_size[2] / 2), math.ceil(r2_size[3] / 2)]
    r4_size = [1, math.ceil(64 * sim_ratio), math.ceil(r3_size[2] / 2), math.ceil(r3_size[3] / 2)]
    r1i = np.zeros(r1_size, dtype=np.float32)
    r2i = np.zeros(r2_size, dtype=np.float32)
    r3i = np.zeros(r3_size, dtype=np.float32)
    r4i = np.zeros(r4_size, dtype=np.float32)
    bgr = np.array([0.14, 0.5, 0.14]).reshape((3, 1, 1))
    dratio = np.array([0.25], dtype=np.float32)

    print(f"Infer {video_path} start ...")
    while video_capture.isOpened():
        success, frame = video_capture.read()

        if success:
            i += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (src_size[3], src_size[2]))  # (w, h)
            src = normalize(frame)
            # src 张量是 [B, C, H, W] 形状，必须用模型一样的 dtype
            t1 = time.time()
            pha, r1o, r2o, r3o, r4o = sess.run(["res", "r1o", "r2o", "r3o", "r4o"], {
                'src': src,
                'r1i': r1i,
                'r2i': r2i,
                'r3i': r3i,
                'r4i': r4i,
            })
            # 更新context
            r1i = r1o
            r2i = r2o
            r3i = r3o
            r4i = r4o

            t2 = time.time()
            print(f"Infer {i}/{frame_count} done! -> cost {(t2 - t1) * 1000} ms", end=" ")
            merge_frame = src * pha + bgr * (1. - pha)  # (1,3,H,W)
            merge_frame = merge_frame[0] * 255.  # (3,H,W)
            merge_frame = merge_frame.astype(np.uint8)  # RGB
            merge_frame = np.transpose(merge_frame, (1, 2, 0))  # (H,W,3)

            # 调整输出的宽高
            merge_frame = cv2.resize(merge_frame, (width, height))

            video_writer.write(merge_frame)
            print(f"write {i}/{frame_count} done.")
        else:
            print("can not read video! skip!")
            break

    video_capture.release()
    video_writer.release()
    print(f"Infer {video_path} done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="video")
    parser.add_argument("--weight", type=str, default="rvm_mobilenetv3_720x1280.onnx")
    parser.add_argument("--input", type=str, default="/data/home/vacingfang/video/meeting_02_720x640.mp4")
    parser.add_argument("--output", type=str, default="m02_onnx.mp4")
    args = parser.parse_args()

    weight = args.weight
    weight = "/data/home/vacingfang/WebInfTest/ort_web/convnets_small_s1_d0.25.onnx "
    weight = "model_seg_sim.onnx"
    if args.mode == "video":
        infer_rvm_video(weight=weight, video_path=args.input, output_path=args.output)
    else:
        infer_rvm_frame(weight=weight, img_path=args.input, output_path=args.output)

    """
    
    PYTHONPATH=. python3 ./inference_onnx_static.py --mode img --input ./res/test.jpg --output ./res/test_onnx.jpg
    """