from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Use prebuild model pre-trained on imagenet

model = VGG16(weights='imagenet', include_top=True)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

im = cv2.resize(cv2.imread('Covnets/cifar_10/train.jpg'), (224, 224))
im = np.expand_dims(im, axis=0)

# predict using pretrained model
out = model.predict(im)
print(np.argmax(out))

plt.plot(out.ravel())