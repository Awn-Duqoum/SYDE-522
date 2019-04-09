import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Load the data
labels = np.load("data\\labels.npy")
samples = np.load("data\\samples.npy")

# Flatten the data
flatSamples = np.zeros((samples.shape[0], samples.shape[1]*samples.shape[2]))
for i in range(samples.shape[0]):
    flatSamples[i] = samples[i].flatten()

# Split the data into testings and training - 10% split
train_img, test_img, train_lbl, test_lbl = train_test_split(flatSamples, labels, test_size=0.1, random_state=None, shuffle=True)

# Let's standardize the data
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# Classify - Logistical Regression
logisticRegr = LogisticRegression(multi_class="multinomial", max_iter=1e9, solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)

# Predict for mean accuracy 
print("LogisticRegression (After PCA) : " + str(logisticRegr.score(test_img, test_lbl)))

test_labels = logisticRegr.predict(test_img)
print(confusion_matrix(test_lbl, test_labels))

# Classify #2 - KNN 
KNNModel = KNeighborsClassifier(n_neighbors=1)
KNNModel.fit(train_img, train_lbl)

# Predict for mean accuracy 
print("KNeighborsClassifier (After PCA) : " + str(KNNModel.score(test_img, test_lbl)))

test_labels = KNNModel.predict(test_img)
print(confusion_matrix(test_lbl, test_labels))