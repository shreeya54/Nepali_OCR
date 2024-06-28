from flask import Flask, render_template, request, url_for
import os
import cv2 as cv
import numpy as np
import base64
import string
from tensorflow.keras.models import load_model
app = Flask(__name__)

# devnagarik_word = '०,१,क,ख,ग,घ,ङ,च,छ,ज,झ,ञ,२,ट,ठ,ड,ढ,ण,त,थ,द,ध,न,३,प,फ,ब,भ,म,य,र,ल,व,श,४,ष,स,ह,क्ष,त्र,ज्ञ,५,६,७,८,९,'
# devnagarik_word = devnagarik_word.split(',')

devnagarik_word = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'क', 'ब', 'भ', 'म', 'य',
                   'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ',
                   '०', '१', '२', '३', '४', '५', '६', '७', '८', '९']

print(devnagarik_word)
print(type(devnagarik_word))

trained_model = load_model('Handwritten_OCR.h5')


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/photo', methods=['GET', 'POST'])
def upload():
    path = "static/outimg"
    imgToRemove = os.listdir(path)
    print(imgToRemove)
    for i in imgToRemove:
        os.remove(path + "/" + i)

    target = os.path.join(APP_ROOT, 'static/images/')
    # print(target)
    # l=[]
    if not os.path.isdir(target):
        os.mkdir(target)

    # for file in request.files.getlist("file"):// for multiple file uplaod
    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    # print(destination)
    file.save(destination)
    newDes = os.path.join('static/images/' + filename)
    readingimg = cv.imread(newDes)

    name = list(string.ascii_letters)
    word = preprocessing(readingimg)
    char = ""
    print(len(word))
    for i in range(len(word)):
        cv.imwrite("static/outimg/image-" + name[i] + ".jpg", word[i])

    total_count = len(word)
    print("*" * 10)
    print("Total letters found: ", total_count)
    probab = 0
    for count, i in enumerate(range(total_count), start=1):
        print("Performing operation for: ", count)
        resize = cv.resize(word[i], (32, 32)) / 255.0
        reshaped = np.reshape(resize, (1, 32, 32, 1))

        prediction = trained_model.predict(reshaped)
        score_prediction = prediction > 0.5
        probab = str(np.amax(prediction))
        max = score_prediction.argmax()
        predict_character = devnagarik_word[max]
        char += predict_character
        print("Predicted character", predict_character)
        print("Predicted character index", max)
        print("Probab", probab)
        print("*" * 10)
    print("End !!")
    print("*" * 10)
    final_char = char

    return render_template('index.html', photos=newDes, result=final_char, probability=probab, processedImg=url_for('static', filename='/outimg/image-a.jpg'),
                           title='NepaliOCR - Predict')

# preprocess image- resize, greyscale, gausianBlur


def ROI(img):
    row, col = img.shape

    np_gray = np.array(img, np.uint8)
    one_row = np.zeros((1, col), np.uint8)

    images_location = []

    line_seg_img = np.array([])
    for r in range(row - 1):
        if np.equal(img[r:(r + 1)], one_row).all():
            if line_seg_img.size == 0:
                current_r = r
            else:
                images_location.append(line_seg_img[:-1])
                line_seg_img = np.array([])
                current_r = r
        else:
            #             print(r)
            if line_seg_img.size <= 1:
                line_seg_img = np.vstack((np_gray[r], np_gray[r + 1]))

            else:
                line_seg_img = np.vstack((line_seg_img, np_gray[r + 1]))

    return images_location


def preprocessing(img):
    # resizing the image
    img = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    image_area = img.shape[0] * img.shape[1]

    #     converting into grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(img_gray, (3, 3), 0)
    _, thresh_img = cv.threshold(
        gaussian, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    #     dilated = cv.dilate(thresh_img, None, iterations=1)

    # finding the boundary of the all threshold images
    contours, _ = cv.findContours(
        thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #     print(len(contours))
    for contour in contours:
        # boundary of each contour
        x, y, w, h = cv.boundingRect(contour)
        # to discard very small noises
        if cv.contourArea(contour) < image_area * 0.0001:
            thresh_img[y:(y + h), x:(x + w)] = 0

    # line segmentation
    line_segmentation = ROI(thresh_img)

    # word segmentation
    each_word_segmentation = []
    for line in np.asarray(line_segmentation):
        word_segementation = ROI(line.T)
        for words in np.asarray(word_segementation):
            each_word_segmentation.append(words.T)

    return each_word_segmentation


if __name__ == "__main__":
    app.run(debug=True)
