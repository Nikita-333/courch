import numpy as np
import tensorflow as tf
import keras
import cv2
import re
import matplotlib.pyplot as plt
from keras import layers
from tensorflow.keras.models import load_model

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


SIZE = 256
high_img = []
path = '/home/rydyar/cours/cur/dataset/Raw Data/high_res'
files = os.listdir(path)
files = sorted_alphanumeric(files)

for i in tqdm(files):
    if i == '855.jpg':
        break
    else:
        img = cv2.imread(path + '/' + i, 1)
        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        high_img.append(img_to_array(img))

low_img = []
path = '/home/rydyar/cours/cur/dataset/Raw Data/low_res'
files = os.listdir(path)
files = sorted_alphanumeric(files)

for i in tqdm(files):
    if i == '855.jpg':
        break
    else:
        img = cv2.imread(path + '/' + i, 1)

        # resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        low_img.append(img_to_array(img))

for i in range(4):
    a = np.random.randint(0,855)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title('High Resolution Image', color = 'green', fontsize = 20)
    plt.imshow(high_img[a])
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title('low Resolution Image ', color = 'black', fontsize = 20)
    plt.imshow(low_img[a])
    plt.axis('off')
plt.show()

train_high_image = high_img[:700]
train_low_image = low_img[:700]
train_high_image = np.reshape(train_high_image, (len(train_high_image), SIZE, SIZE, 3))
train_low_image = np.reshape(train_low_image, (len(train_low_image), SIZE, SIZE, 3))

validation_high_image = high_img[700:830]
validation_low_image = low_img[700:830]
validation_high_image = np.reshape(validation_high_image, (len(validation_high_image), SIZE, SIZE, 3))
validation_low_image = np.reshape(validation_low_image, (len(validation_low_image), SIZE, SIZE, 3))

test_high_image = high_img[830:]
test_low_image = low_img[830:]
test_high_image = np.reshape(test_high_image, (len(test_high_image), SIZE, SIZE, 3))
test_low_image = np.reshape(test_low_image, (len(test_low_image), SIZE, SIZE, 3))

print("Shape of training images:", train_high_image.shape)
print("Shape of test images:", test_high_image.shape)
print("Shape of validation images:", validation_high_image.shape)

def down(filters, kernel_size, apply_batch_normalization=True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters, kernel_size, padding='same', strides=2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(keras.layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout=False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size, padding='same', strides=2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add(keras.layers.LeakyReLU())
    return upsample


def model():
    inputs = layers.Input(shape=[SIZE, SIZE, 3])
    d1 = down(128, (3, 3), False)(inputs)
    d2 = down(128, (3, 3), False)(d1)
    d3 = down(256, (3, 3), True)(d2)
    d4 = down(512, (3, 3), True)(d3)

    d5 = down(512, (3, 3), True)(d4)

    u1 = up(512, (3, 3), False)(d5)
    u1 = layers.concatenate([u1, d4])
    u2 = up(256, (3, 3), False)(u1)
    u2 = layers.concatenate([u2, d3])
    u3 = up(128, (3, 3), False)(u2)
    u3 = layers.concatenate([u3, d2])
    u4 = up(128, (3, 3), False)(u3)
    u4 = layers.concatenate([u4, d1])
    u5 = up(3, (3, 3), False)(u4)
    u5 = layers.concatenate([u5, inputs])
    output = layers.Conv2D(3, (2, 2), strides=1, padding='same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)


model = model()
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error',
              metrics=['acc'])

model.fit(train_low_image, train_high_image, epochs=7, batch_size=1,
          validation_data=(validation_low_image, validation_high_image))


def plot_images(high, low, predicted):
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('High Image', color='green', fontsize=20)
    plt.imshow(high)
    plt.subplot(1, 3, 2)
    plt.title('Low Image ', color='black', fontsize=20)
    plt.imshow(low)
    plt.subplot(1, 3, 3)
    plt.title('Predicted Image ', color='Red', fontsize=20)
    plt.imshow(predicted)

    plt.show()


for i in range(1, 10):
    predicted = np.clip(model.predict(test_low_image[i].reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(SIZE, SIZE, 3)
    plot_images(test_high_image[i], test_low_image[i], predicted)

model.save("final_model.h5")
#
#
model = load_model('final_model.h5')


img = cv2.imread('image/0001.png', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


SIZE = 256
img = cv2.resize(img, (SIZE, SIZE))


img = img.astype('float32') / 255.0


predicted = model.predict(img.reshape(1, SIZE, SIZE, 3))


predicted = np.clip(predicted, 0.0, 1.0).reshape(SIZE, SIZE, 3)


predicted = (predicted * 255).astype('uint8')

#predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)


cv2.imwrite('image1/0001.png', predicted)
