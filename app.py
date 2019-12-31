import numpy as np
import torch
import os
import time
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
# Keras
from PIL import Image
from torchvision import transforms

from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

import os

model_ft=torch.load('models/model')
app = Flask(__name__)
dropzone = Dropzone(app)


app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
app.config['DROPZONE_DEFAULT_MESSAGE'] = 'Arrastra tu foto hasta aqui'

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


@app.route('/', methods=['GET', 'POST'])
def index():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']

    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            
            # save the file with to our photos folder
            filename = photos.save(
                file,
                name=file.filename    
            )
            
            prediction=model_predict('./uploads/'+filename)
            
            session['my_object']=prediction
            # append image urls
            file_urls.append(photos.url(filename))
            
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request    
    return render_template('index3.html')


@app.route('/results')
def results():
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('index'))
    
    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    
    prediction=session.get('my_object')
    
    session.pop('file_urls', None)
   
    return render_template('results.html', file_urls=file_urls,predictions=prediction)

def model_predict(img_path):
    # load image with target size

    input_image = Image.open(img_path)
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


    with torch.no_grad():
        output = model_ft(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        #print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    preds=torch.nn.functional.softmax(output[0], dim=0).tolist()
    category_qual=max(zip(preds, range(len(preds))))[1]
    if category_qual==0:
        message_qual="Puede mejorar!"
    elif category_qual==1:
         message_qual="Inmejorable, súbela!"
    elif category_qual==2:
        message_qual="Intenta con otra!"
    return message_qual

if __name__ == '__main__':
    app.run()