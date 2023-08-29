
# python libraries---->

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np


app = Flask(__name__)


dic = {0: 'NORMAL', 1: 'PNEUMONIA'}


try:
    model = load_model('model.h5')
    model.make_predict_function()
except:
    pass


def predict_label(img_path):

  i = image.load_img(img_path,target_size=(156,156))
  i = image.img_to_array(i)/255.0
  i = i.reshape(1,156,156,3)
  pre= model.predict(i)
  predicted_class_index = np.argmax(pre)
  p = dic[predicted_class_index]

  return p


@app.route('/', methods=['GET','POST'])

def main():
     return render_template('p1.html')


@app.route("/submit", methods=['GET','POST'])
def result():

  if request.method == 'POST':

    img=request.files['images']  
    img_path="static/" + img.filename
    img.save(img_path)
    p = predict_label(img_path)

  return render_template("p1.html", prediction = p, img_path=img_path)


if __name__=="__main__":

    app.run(debug = True, port = 7000)


