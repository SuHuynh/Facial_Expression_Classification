import torch
from vgg16 import VGG16
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np

classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Comtempt']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Define hyper-parameter
img_size = (48, 48)

#define model
model = VGG16()
pre_trained = torch.load('./saved_models/saved_model_epoch_20.pth')
model.load_state_dict(pre_trained)

#port to model to gpu if you have gpu
model = model.to(device)
model.eval()

# load and pre-process testing image
# Note: you need to precess testing image similarly to the training images 
img_path = './test_img.jpg'
img_raw = cv2.imread(img_path)

# resize img to 48x48
img = cv2.resize(img_raw, img_size)

# convert from RGB img to gray img
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# normalize img from [0, 255] to [0, 1]
img_gray = img_gray/255
img_gray = img_gray.astype('float32')

# convert image to torch with size (1, 1, 48, 48)
img_gray = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    img_gray = img_gray.to(device)
    print(type(img_gray))
    y_pred = model(img_gray)           
    _, pred = torch.max(y_pred, 1)
    pred = pred.data.cpu().numpy()
    print(pred)
    emotion_prediction = classes[pred[0]]

    cv2.putText(img_raw, 'Predict: '+emotion_prediction, (20, 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
    cv2.imshow(emotion_prediction, np.array(img_raw))
    cv2.waitKey(0)



