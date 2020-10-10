from dataset_explore import import_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern
# from skimage.feature import corner_fast
# from skimage import feature
from skimage.feature import hog
# from skimage import io
# from matplotlib import pyplot as plt
from tempfile import TemporaryFile
trained_feat = TemporaryFile()
tested_feat = TemporaryFile()

def get_features():
    X_train, X_test, y_train, y_test = import_data()
    fd = []
   # X_train_lbp = np.copy(X_train)
   # X_test_lbp = np.copy(X_test)
    fd_feat_train = np.ones(324)
    fd_feat_test = np.ones(324)
    for ind, image in enumerate(X_train):
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        fd_feat_train = np.column_stack((fd_feat_train, fd))

         # X_train[ind] = gaussian_filter(image, sigma=0.18)
         # filtered_coords = corner_fast(X_train)
        # X_train[ind] = feature.canny(X_train[ind])
        #   X_train_lbp[ind] = local_binary_pattern(image, 5, 5)
    fd_feat_train = fd_feat_train[:, 1:]
    fd_feat_train = np.transpose(fd_feat_train)
    for ind, image in enumerate(X_test):
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(5, 5),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        fd_feat_test = np.column_stack((fd_feat_test, fd))
       # X_test[ind] = gaussian_filter(image, sigma=0.18)
       # X_test_lbp[ind] = local_binary_pattern(image, 5, 5)
        # X_test[ind] = feature.canny(X_test[ind])
    fd_feat_test = fd_feat_test[:, 1:]
    fd_feat_test = np.transpose(fd_feat_test)

   #  X_train = X_train.reshape(X_train.shape[0], 400)
   # X_test = X_test.reshape(X_test.shape[0], 400)
   # X_train_lbp = X_train_lbp.reshape(X_train.shape[0], 400)
   # X_train = np.append(X_train, X_train_lbp, axis=1)
   # scaler = StandardScaler().fit(X_train)
   # X_train = scaler.transform(X_train)
   # X_train_lbp = X_train_lbp.reshape(X_train.shape[0], 400)
   # X_train = np.append(X_train, X_train_lbp, axis=1)
   # print("mean = ", np.mean(X_train))
   # print("deviation = ", np.var(X_train))
  #  X_test_lbp = X_test_lbp.reshape(X_test.shape[0], 400)
  #  X_test = np.append(X_test, X_test_lbp, axis=1)
   # X_test = scaler.transform(X_test)

    np.save(trained_feat, fd_feat_train)
    np.save(tested_feat, fd_feat_test)

    # Principle Component Analysis

    X_pca = PCA(n_components=0.8)
    X_pca.fit(fd_feat_train)
    fd_feat_train_reduced = X_pca.transform(fd_feat_train)
    fd_feat_test_reduced = X_pca.transform(fd_feat_test)
    return fd_feat_train_reduced, fd_feat_test_reduced, y_train, y_test
    # return X_train_reduced, X_test_reduced, y_train, y_test
# feature_vectors = X_train_pca.fit_transform(X_train)

# X_train_reduced, X_test_reduced, y_train, y_test = get_features()

# print(len(y_train))
# plt.imshow(X_train[2323], cmap='hot')
# print(y_train[2323])
# print(X_train[2323])
# plt.show()
