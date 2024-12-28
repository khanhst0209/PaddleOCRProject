from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2

# Khởi tạo OCR với mô hình tùy chỉnh
ocr = PaddleOCR(
    det_model_dir='path_to_your_detection_model',  # Mô hình phát hiện tùy chỉnh
    rec_model_dir='path_to_your_recognition_model',  # Mô hình nhận diện tùy chỉnh
    use_gpu=True,
    lang='en'
)

# Đường dẫn tới ảnh
image_path = 'path_to_your_image.jpg'

# Nhận diện văn bản
result = ocr.ocr(image_path, det=True, rec=True)

# In kết quả nhận diện
for line in result[0]:
    print(f"Detected text: {line[1][0]}, Confidence: {line[1][1]}")

# Vẽ bounding box và văn bản lên ảnh
image = Image.open(image_path).convert('RGB')
boxes = [line[0] for line in result[0]]  # Lấy tọa độ bounding box
texts = [line[1][0] for line in result[0]]  # Lấy văn bản nhận diện
scores = [line[1][1] for line in result[0]]  # Lấy độ tin cậy

# Tạo ảnh kết quả
result_image = draw_ocr(image, boxes, texts, scores, font_path='path_to_font.ttf')

# Hiển thị và lưu ảnh kết quả
result_image = Image.fromarray(result_image)
result_image.show()  # Hiển thị ảnh
result_image.save('ocr_result.jpg')  # Lưu ảnh kết quả
