import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# 初始化 Mediapipe 的人脸关键点检测模型
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

# 创建一个用于显示图像的 Streamlit 窗口
st.title("人脸关键点检测")

# 自定义 VideoTransformer 类，用于处理每一帧图像
class MediapipeTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()

    def transform(self, frame):
        # 将帧转换为 RGB 格式
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 调用 Mediapipe 进行人脸关键点检测
        results = mp_face_mesh.process(img_rgb)

        # 获取检测结果中的人脸关键点
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    # 在图像上绘制人脸关键点
                    height, width, _ = img.shape
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # 将图像数据转换为 RGB 格式
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img_bgr

# 创建一个 Streamlit-WebRTC 组件，并指定使用自定义的 VideoTransformer 类
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=MediapipeTransformer)

# 创建一个侧边栏，用于选择图像输入方式
option = st.sidebar.selectbox(
    "How would you like to input the picture?",
    ("Upload", "Camera", "Real-time")
)

if option == "Upload":
    picture = st.file_uploader("Choose a file")
    if picture:
        st.image(picture)

if option == "Camera":
    picture = st.camera_input("Take a picture")
    if picture:
        st.image(picture)

if option == "Real-time":
    if webrtc_ctx.video_transformer:
        if st.button("Stop"):
            webrtc_ctx.video_transformer = None
    else:
        if st.button("Start"):
            webrtc_ctx.video_transformer = MediapipeTransformer()