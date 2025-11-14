import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 加载预训练模型
model = load_model('face_classification_model.h5')

def recognize_face_offline(image_path, model):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 预处理图像：调整大小、归一化、添加通道维度
    img_resized = cv2.resize(gray, (150, 150))  # 修改为150x150以匹配模型输入
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=-1)
    img_input = np.expand_dims(img_input, axis=0)

    # 进行预测
    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence

# 测试单张图像
test_image_path = 'test.jpg'  # 替换为实际测试图像路径
try:
    predicted_class, confidence = recognize_face_offline(test_image_path, model)
    print(f'\n离线测试结果：')
    print(f'识别身份：{predicted_class}')
    print(f'置信度：{confidence:.4f}')
except Exception as e:
    print(f"发生错误: {e}")
