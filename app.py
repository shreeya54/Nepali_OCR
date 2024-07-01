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
def upload_file():
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


def dikka_remove(output):  # Needed for Word segmentation
    resultafterdikka = []
    each_character = []

    for i in range(0, len(output)):
        each = []
        main = output[i]
        r, inv3 = cv.threshold(main, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        ig = output[i]
        row, col = ig.shape

 # Detects and removes the largest horizontal line (referred to as "DIKA") in an image
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
        detect_horizontal = cv.morphologyEx(
            ig, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts, _ = cv.findContours(
            detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)
            X, Y, w, h = cv.boundingRect(c)
            ig[0:Y + h + 2, 0:X + w].fill(0)

            r, inv1 = cv.threshold(
                ig, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            cnts1, _ = cv.findContours(
                inv1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for co in reversed(cnts1):
                if cv.contourArea(co) > 300:
                    X, Y, w, h = cv.boundingRect(co)
                    cv.rectangle(inv3, (X, 0), (X + w, Y + h), 255, 1)
                    each.append((inv3[0:Y + h, X:X + w]))
            each_character.append(each)
            resultafterdikka.append(inv3)

    return resultafterdikka, each_character


# @app.route('/cleanup')
# def cleanup_file():
#     path = "static/outimg"
#     willremoveimage = os.listdir(path)
#     if not willremoveimage:
#         pass
#     else:
#         for i in willremoveimage:
#             os.remove(path + "/" + i)  #For each file,deletes the file from the directory.

#     return render_template('index.html', title='Devnagarik - Home')


@app.route('/send_pic', methods=['POST'])
def button_pressed():  # letter detect and predict
    # print("Image recieved")
    dimensions = (100, 100)
    data_url = request.values['imgBase64']
    encoded_data = data_url.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    name = list(string.ascii_letters)
    word = preprocessing(img)
    char = ""
    for i in range(len(word)):
        cv.imwrite("static/outimg/image-" + name[i] + ".jpg", word[i])

    total_count = len(word)
    print("*" * 10)
    print("Total letters found: ", total_count)
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
    return char


@app.route('/uploader', methods=['GET', 'POST'])
def upload():  # Word detect and predict
    path = "static/outimg"
    willremoveimage = os.listdir(path)
    print(willremoveimage)
    for i in willremoveimage:
        os.remove(path + "/" + i)

    target = os.path.join(APP_ROOT, 'static/images/')
    # print(target)
    # l=[]
    if not os.path.isdir(target):
        os.mkdir(target)

    if request.method == 'POST':
        # for file in request.files.getlist("file"):// for multiple file uplaod
        file = request.files['file']
        filename = file.filename
        destination = "/".join([target, filename])
        # print(destination)
        file.save(destination)
        newDes = os.path.join('static/images/' + filename)
        readingimg = cv.imread(newDes)

        def prediction(each_character):
            final_all_word = ""
            prob = 0
            for i in range(len(each_character)):
                each_word = ""
                for j in each_character[i]:
                    character_img = j
                    resize = cv.resize(j, (32, 32)) / 255.0
                    reshaped = np.reshape(resize, (1, 32, 32, 1))
                    prediction = trained_model.predict(reshaped)
                    prob = prob + np.amax(prediction)
                    print("Prob=", prob)
                    max = prediction.argmax()
                    predict_character = devnagarik_word[max]
                    each_word += predict_character
                final_all_word += each_word + ' '
            return final_all_word, prob

        output = preprocessing(readingimg)
        resultafterdikka, each_character = dikka_remove(output)
        final_result, prob = prediction(each_character)
        prob = prob/len(each_character[0])
        print("Final Prob=", prob)

        makingimagename = list(string.ascii_letters)

        for i in range(len(resultafterdikka)):
            cv.imwrite("static/outimg/image-" +
                       makingimagename[i] + ".jpg", resultafterdikka[i])

        images = os.listdir("static/outimg")

    return render_template('index.html', photos=newDes, images_name=images, result=final_result, processedImg=url_for('static', filename='/outimg/image-a.jpg'), probability=prob, title='Devnagrik - Predict')


if __name__ == '__main__':
    app.run(debug=True)