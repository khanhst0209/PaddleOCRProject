from flask import Flask, request, jsonify
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

app = Flask(__name__)

# Khởi tạo OCR với mô hình tùy chỉnh
ocr = PaddleOCR(
    det_model_dir='models/det',
    rec_model_dir='models/rec2',
    use_gpu=True,
)

@app.route('/ocr', methods=['POST'])
def ocr_process():
    try:
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Lấy file ảnh từ request
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')

        # Nhận diện văn bản
        result = ocr.ocr(np.array(image), det=True, rec=True)

        # Lấy tọa độ, văn bản, và độ tin cậy
        boxes = [line[0] for line in result[0]]
        texts = [line[1][0] for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        # Vẽ bounding box và văn bản lên ảnh
        result_image = draw_ocr(image, boxes, texts, scores, font_path='NomNaTong-Regular.ttf')

        # Chuyển đổi ảnh kết quả sang Base64 để trả về
        buffered = BytesIO()
        Image.fromarray(result_image).save(buffered, format="JPEG")
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Trả về kết quả JSON
        return jsonify({
            'text_results': [{'text': text, 'confidence': score} for text, score in zip(texts, scores)],
            'image': result_image_base64  # Ảnh trả về dạng Base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
