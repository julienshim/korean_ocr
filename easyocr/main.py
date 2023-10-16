import easyocr
import cv2
from matplotlib import pyplot as plt

reader = easyocr.Reader(['ko', 'en'])
image_path = '../test_data/jpeg/korean-grammar-in-use-beginner/korean-grammar-in-use-beginner-186.jpg'
result = reader.readtext(image_path)



img = cv2.imread(image_path)

for r in result:
    top_left = list(map(lambda x: int(x), r[0][0]))
    bottom_right = list(map(lambda x: int(x), r[0][2]))
    text = r[1] 
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 5)
    img = cv2.putText(img, text, top_left, font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

plt.imshow(img)
plt.show()
