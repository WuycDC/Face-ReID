import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载预训练模型
model = load_model('face_classification_model.h5')

# 加载Haar级联分类器进行人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # 裁剪人脸区域
        face_gray = gray[y:y+h, x:x+w]

        # 预处理：调整大小、归一化、添加通道维度
        face_resized = cv2.resize(face_gray, (150, 150))  # 修改为150x150
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=-1)
        face_input = np.expand_dims(face_input, axis=0)

        # 进行预测
        prediction = model.predict(face_input)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100

        # 在图像上绘制识别结果
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f'Person {predicted_class}: {confidence:.2f}%'
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示图像
    cv2.namedWindow('Face-REid', cv2.WINDOW_NORMAL)
    cv2.imshow('Face-REid', frame)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()


