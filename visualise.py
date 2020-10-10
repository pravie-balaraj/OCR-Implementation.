from dataset_explore import import_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern
from skimage import feature
from skimage import io
from skimage.feature import hog
from skimage.feature import corner_harris,  corner_peaks
from matplotlib import pyplot as plt


X_train, X_test, y_train, y_test = import_data()
fd_feat_train = np.ones(324)
fd_feat_test = np.ones(324)
for ind, image in enumerate(X_train):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
    fd_feat = np.column_stack((fd_feat_train, fd))

    for indi, image in enumerate(X_train):
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        fd_feat_test = np.column_stack((fd_feat_test, fd))
    # X_train[ind] = gaussian_filter(image, sigma=0.15)
    # corners = corner_peaks(corner_harris(X_train[ind]), min_distance=1)
    # X_train[ind] = local_binary_pattern(image, 5, 5)
    # X_train[ind] = feature.canny(X_train[ind], sigma=1)
#for ind, image in enumerate(X_test):
    # X_test[ind] = gaussian_filter(image, sigma=4)
#    X_test[ind] = feature.canny(X_test[ind], sigma=3)

# print(len(corners))
fd_feat_train = fd_feat_train[:, 1:]
fd_feat_train = np.transpose(fd_feat_train)
fd_feat_test = fd_feat_test[:, 1:]
fd_feat_test = np.transpose(fd_feat_test)

print("fd_train_shape=", fd_feat_train.shape)
print("fd_test_shape=", fd_feat_test.shape)
print("y_train_shape=", y_train.shape)
print("y_test_shape=", y_test.shape)
# io.imshow(X_train[2321])
# print(y_train[2321])
# io.show()

# X_train = X_train.reshape(X_train.shape[0], 400)
# X_test = X_test.reshape(X_test.shape[0], 400)
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# print("mean = ", np.mean(X_train))
# print("deviation = ", np.var(X_train))
# X_test = scaler.transform(X_test)
#
# # Display Image
#
# # io.imshow(X_train[2321])
# # print(y_train[2321])
# # io.show()
# # Principle Component Analysis
#
# X_pca = PCA(n_components=0.9)
# X_pca.fit(X_train)
# X_train_reduced = X_pca.transform(X_train)
# X_test_reduced = X_pca.transform(X_test)
#





