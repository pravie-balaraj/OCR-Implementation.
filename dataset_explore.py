from skimage import io
import os
import numpy as np
from sklearn.model_selection import train_test_split


def import_data():

    image_dir = 'chars74k-lite'
    data_label = []
    data_image = []
    image_sub_dirs = os.listdir(image_dir)

    for image_sub_dir in image_sub_dirs:
        if image_sub_dir != 'LICENSE':
            path_to_images = os.path.join(image_dir, image_sub_dir)
            images = os.listdir(path_to_images)
            for image in images:
                image = os.path.join(path_to_images, image)
                img = io.imread(image, format='jpg')
                img = np.array(img, dtype=np.float32)       # Images have been already scaled while reading
                img /= 255                                     # EASIER FOR EXTRACTION
                data_image.append(img)
                data_label.append(ord(image_sub_dir)-ord('a'))

    data_image = np.array(data_image)
    data_label = np.array(data_label)
    X_train, X_test, y_train, y_test = train_test_split(data_image, data_label, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = import_data()
# print(len(y_train))
# io.imshow((X_train[2323]))
# print(y_train[2323])
# io.show()
#
# io.imshow((X_test[323]))
# print(y_test[323])
# io.show()


# split_folders.ratio('chars74k-lite', ratio=(0.8, 0.2))

# print(image_labels)
# image_sub_dir = image_sub_dirs[5]
# print(image_sub_dir)
# path_to_images = os.path.join(image_dir, image_sub_dir)
# images = os.listdir(path_to_images)
# image = images[0]
# image = os.path.join(path_to_images, image)
# img = io.imread('a_1.jpg', format='jpg')
# io.imshow(img)

# plt.plot(img)
# io.show()