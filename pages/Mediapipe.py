import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# 初始化 Mediapipe 的人脸关键点检测模型
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

# 创建一个用于显示图像的 Streamlit 窗口
st.title("人脸关键点检测")

# 使用 Streamlit 的 camera_input 函数获取摄像头图像
img_file_buffer = st.camera_input("拍照")

if img_file_buffer is not None:
    # 将图像文件缓冲区的数据转换为 OpenCV 中的图像格式
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # 将图像转换为 RGB 格式
    cv2_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    # 调用 Mediapipe 进行人脸关键点检测
    results = mp_face_mesh.process(cv2_rgb)
    
    # 获取检测结果中的人脸关键点
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # 在图像上绘制人脸关键点
                height, width, _ = cv2_img.shape
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(cv2_img, (x, y), 2, (0, 255, 0), -1)
    
    # 将图像数据转换为字节流
    cv2_img_bytes = cv2.imencode('.jpg', cv2_img)[1].tobytes()
    
    # 使用 Streamlit 显示图像
    st.image(cv2_img_bytes, channels="BGR")