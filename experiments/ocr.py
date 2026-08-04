import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

img = cv2.imread("test.png")

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image,5)

def thresholding(image):
    return cv2.threshold(image, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def ocr_core(img):
    return pytesseract.image_to_string(img, config='--oem 3 --psm 6')

img = get_grayscale(img)
img = thresholding(img)
img = remove_noise(img)

print(ocr_core(img))