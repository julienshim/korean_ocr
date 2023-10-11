from PIL import Image, ImageFilter, ImageDraw
from tesserocr import PyTessBaseAPI, get_languages, tesseract_version, RIL, image_to_text, file_to_text

print(tesseract_version()) # print tesseract-ocr version
print(get_languages('./tessdata')) # prints tessdata path and list of available languages

images = ['../test_data/jpeg/korean-grammar-in-use-beginner/korean-grammar-in-use-beginner-186.jpg']

with PyTessBaseAPI(path='./tessdata/', lang='kor+eng') as api:
    for img in images:
        with Image.open(img) as test_image:
            test_image = test_image.filter(ImageFilter.UnsharpMask())
            new_size = (test_image.size[0] * 2, test_image.size[1] * 2)
            test_image = test_image.resize(new_size)
            api.SetImage(test_image)
            test_image.show()

            utf8_text = api.GetUTF8Text()
            all_world_confidences = api.AllWordConfidences()

