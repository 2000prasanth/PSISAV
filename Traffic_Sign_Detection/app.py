from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import uuid

os.chdir('/Users/cs/Desktop/GTSRB/Traffic_Sign_Detection')
classes = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing veh over 3.5 tons',
    11:'Right-of-way at intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Vehicle > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing vehicle > 3.5 tons'
}

model = load_model('/Users/cs/Desktop/GTSRB/Traffic_Sign_Detection/my_model.h5')

def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

def equalize(img):
  img = cv2.equalizeHist(img)
  return img

def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  #normalize the images, i.e. convert the pixel values to fit btwn 0 and 1
  img = img/255
  return img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'
#app.config['UPLOAD FOLDER']='/Users/cs/Desktop/GTSRB/Traffic_Sign_Detection/static/'

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/uploader" , methods=['GET', 'POST'])
def uploader():    
    if request.method=='POST':
        f = request.files['file1']
        #filename = str(uuid.uuid4()) + ".jpg" 
        filename=secure_filename(f.filename)
        #f.filename = "image.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)
        #f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        #img = Image.open("static/image.jpg")
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
        img = np.asarray(img)
        img = cv2.resize(img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        result = classes[int(str(np.argmax(model.predict(img), axis=-1)[0]))]
        #pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
        pic1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Use the saved filename
        return render_template("uploaded.html", sign_name=result, input_image=file_path)

if __name__ == '__main__':
    app.run(debug=True,port=6101) 
