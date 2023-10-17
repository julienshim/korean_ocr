import easyocr
import cv2
from matplotlib import pyplot as plt
from os import listdir, path

reader = easyocr.Reader(['ko', 'en'])
test_folder = '../test_data/jpeg/'
image_paths = list(filter(lambda x: x.endswith('.jpg'), [i for i in listdir(test_folder)]))
image_paths = list(map(lambda x: path.join(test_folder, x), image_paths))

for image_path in image_paths:
    image_path = image_path.replace(' ', '\ ')
    result = reader.readtext(image_path)

    img = cv2.imread(image_path)

    for r in result:
        # top_left = list(map(lambda x: int(x), r[0][0]))
        # bottom_right = list(map(lambda x: int(x), r[0][2]))
        text = r[1] 
        print(text)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 5)
        # img = cv2.putText(img, text, top_left, font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    plt.imshow(img)
    plt.show()
