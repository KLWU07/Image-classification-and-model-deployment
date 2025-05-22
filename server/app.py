import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import json
import numpy as np
from model import AlexNet  # 确保您的模型定义可用

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载模型和类别信息
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet(num_classes=5, init_weights=True)
model.load_state_dict(torch.load('model/AlexNet-flower.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()

# 加载类别信息
with open('model/class_indices.json', 'r') as f:
    class_indices = json.load(f)

# 图像预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 预处理图像
            img = Image.open(filepath)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            # 预测
            with torch.no_grad():
                output = model(img.to(device))
                predict = torch.softmax(output, dim=1)
                prob, classes = torch.max(predict, 1)

            class_name = class_indices[str(classes.item())]
            confidence = prob.item()

            # 清理上传的文件
            os.remove(filepath)

            return jsonify({
                'class': class_name,
                'confidence': confidence,
                'class_id': classes.item()
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    # 确保上传文件夹存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)