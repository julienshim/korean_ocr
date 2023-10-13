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
            # api.SetImage(test_image)
            # test_image.show()

            # utf8_text = api.GetUTF8Text()
            # all_world_confidences = api.AllWordConfidences()

            # print(utf8_text)
            # print(all_world_confidences)

            api.SetImage(test_image)
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)
            print('Found {} textline image components.'.format(len(boxes)))

            draw = ImageDraw.Draw(test_image)

            for i, (im, box, _, _) in enumerate(boxes):
                # im is a PIL image object
                # box is a dict with x, y, w and h keys
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])

                draw.rectangle((box['x'], box['y'], box['x']+ box['w'], box['y'] + box['h']), outline='red', width=5)

                ocrResult = api.GetUTF8Text()
                conf = api.MeanTextConf()
                print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                    "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))
                
            test_image.show()

