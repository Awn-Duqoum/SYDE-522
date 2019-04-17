import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

class SKLearnModel(): 
    def __init__(self, name, model):
        self.name = name
        self.model = model
        
    def score(self, data, labels):
        return (self.grid.score(data,labels))
        
    def predict(self, data):
        return(self.grid.predict(data))
        
    def fit(self, data, labels):
        param_grid = {'C'    : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        grid = GridSearchCV(self.model, param_grid, cv=5)
        grid.fit(data, labels)
        self.grid = grid.best_estimator_
        return()

def MovingAverage(X, window = 10):
  for i in range(X.shape[0]):
    for j in range(X.shape[2]):
      for k in range(X.shape[1] - window):
        X[i,k,j] = np.sum(X[i,k:k+window+1,j])/window
  return X
        
def SplitData(numClasses, data, labels, splitRatio):
    # Makes the first split
    train_img_i, test_img_i, train_lbl_i, test_lbl_i = train_test_split(data[labels==0], labels[labels==0], test_size=splitRatio, random_state=None, shuffle=True)
    train_img = train_img_i
    test_img  = test_img_i
    train_lbl = train_lbl_i
    test_lbl  = test_lbl_i
    
    for ii in range(numClasses):
        if(ii == 0):
            continue
        train_img_i, test_img_i, train_lbl_i, test_lbl_i = train_test_split(data[labels==ii], labels[labels==ii], test_size=splitRatio, random_state=None, shuffle=True)
        train_img = np.concatenate((train_img, train_img_i))
        test_img  = np.concatenate((test_img, test_img_i))
        train_lbl = np.concatenate((train_lbl, train_lbl_i))
        test_lbl  = np.concatenate((test_lbl, test_lbl_i))    
    return train_img, test_img, train_lbl, test_lbl
        
# Control Output
OutputConfusionMatrix = False
NumberofFolds = 5
errors = []

# Create Models
logisticRegr = LogisticRegression(multi_class="multinomial", max_iter=1e9, solver = 'lbfgs')
KNNModel = KNeighborsClassifier(n_neighbors=1)
clf = svm.SVC(gamma='scale', decision_function_shape='ovo', verbose=0)
mlp = MLPClassifier(hidden_layer_sizes=(64,128, 256), max_iter=450, alpha=1e-4,
                    solver='adam', tol=1e-8, random_state=1, early_stopping=True,
                    learning_rate_init=0.001, verbose=0)

# Create Model array
models = [SKLearnModel("SVM",clf)]

# Define augmentation parameters
acc_Noise  = 0.02
gyro_Noise = 10
          
# Load the data
labels = np.load("data\\labels.npy")
samples = np.load("data\\samples.npy")

# Determine the number of classes
numClasses = len(np.unique(labels))

# Shuffle the data
newIDX = np.random.permutation(len(samples))
samples = samples[newIDX]
labels = labels[newIDX]

# Smooth the data
samples = MovingAverage(samples, 10)

# Convert Degrees to Radians
samples[:,:,3:6] = samples[:,:,3:6] * np.pi / 360.0

# Subtract out the mean of the data
for i in range(6):
  samples[:,:,i] = samples[:,:,i] - np.mean(samples[:,:,i])

# Create a list to store the errors
errors = np.zeros((len(models), NumberofFolds))

for i in range(NumberofFolds):
    # Update the user
    print(str(i) + ", ", end="")
    
    # Flatten the data
    flatSamples = np.zeros((samples.shape[0], samples.shape[1]*samples.shape[2]))
        
    for ii in range(samples.shape[0]):
        flatSamples[ii] = samples[ii].flatten()

    # Split the data into testings and training - 20% split
    # 20% might get us in a case where we don't have one of the classes so instead of taking 20% of the entire space, take 20% of each class and then merge them
    train_img, test_img, train_lbl, test_lbl = SplitData(numClasses, flatSamples, labels, 0.2)

    for iii in range(3):
        # Create augmented data
        Aug_Samples = train_img.copy()
        Aug_Samples = Aug_Samples.reshape(train_img.shape[0], samples.shape[1], samples.shape[2])
        Aug_Samples[:,:,0:3] = Aug_Samples[:,:,0:3] + np.random.normal(0.0, acc_Noise , size=(Aug_Samples.shape[0], Aug_Samples.shape[1], 3))
        Aug_Samples[:,:,3:6] = Aug_Samples[:,:,3:6] + np.random.normal(0.0, gyro_Noise, size=(Aug_Samples.shape[0], Aug_Samples.shape[1], 3))
        
        # Flatten the data
        flatSamples = np.zeros((Aug_Samples.shape[0], Aug_Samples.shape[1]*Aug_Samples.shape[2]))
        
        for ii in range(Aug_Samples.shape[0]):
            flatSamples[ii] = Aug_Samples[ii].flatten()
            
        # Concatenate the training data and the augmented data
        train_img = np.concatenate((train_img, flatSamples), axis=0)
        train_lbl = np.concatenate((train_lbl,   train_lbl), axis=0)
    
    # Let's standardize the data
    scaler = MaxAbsScaler()
    
    # Fit on training set only.
    scaler.fit(train_img)
    
    # Apply transform to both the training set and the test set.
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img) 
    
    # Create a PCA model with 0.98% variance
    pca = PCA(0.98)
    pca.fit(train_img)
    
    # Transform the data
    train_img = pca.transform(train_img)
    test_img = pca.transform(test_img)
    
    # Score each model
    for ii in range(len(models)):
        # Train the model
        models[ii].fit(train_img, train_lbl)
        
        # Run the model on test data  
        errors[ii, i] = models[ii].score(test_img, test_lbl)

# Output all the errors
for j in range(len(models)):
    print(['{:.2f}'.format(ii) for ii in errors[j]], end=" - ")
    print('{:.2f}'.format(np.mean(errors[j])), end=" - ")
    print(models[j].name)
