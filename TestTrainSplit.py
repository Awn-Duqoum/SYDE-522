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
from scipy import ndimage

# Flatten Data 
def flattenData(X):
    # Create a structure to fold the data
    flatX = np.zeros((X.shape[0], X.shape[1]*X.shape[2]))
    # For each axis flatten
    for ii in range(X.shape[0]):
        flatX[ii] = X[ii].flatten()
        
    return flatX 

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

class SKLearnModel(): 
    def __init__(self, name, model):
        self.name = name
        self.model = model
        
    def score(self, data, labels):
        return (self.model.score(data,labels))
        
    def predict(self, data):
        return(self.model.predict(data))
        
    def fit(self, data, labels):
        return(self.model.fit(data, labels))

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
        
def MovingAverage(X, window = 10):
  # Compute the moving average for the bulk of the signal
  for i in range(X.shape[0]):
    for j in range(X.shape[2]):
      for k in range(X.shape[1] - window):
        X[i,k,j] = np.sum(X[i,k:k+window+1,j])/window
   # For the tail of the data we can't compute a moving average 
   # so just use the mean for it
  for i in range(X.shape[0]):
    for j in range(X.shape[2]):
      X[i,X.shape[1]-window:X.shape[1]+1,j] = X[i, X.shape[1]-window-1, j]
  return X

# Control Output
OutputConfusionMatrix = False
NumberofFolds = 5
errors = []

# Create Models
logisticRegr = LogisticRegression(multi_class="multinomial", max_iter=1e9, solver = 'lbfgs')
KNNModel = KNeighborsClassifier(n_neighbors=1)
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
mlp = MLPClassifier(hidden_layer_sizes=(64, 128, 256), max_iter=450, alpha=1e-4,
                    solver='adam', tol=1e-8, random_state=1, early_stopping=True,
                    learning_rate_init=0.001, verbose=0)

# Create Model array
models = [SKLearnModel("Logistic Regression", logisticRegr), 
          SKLearnModel("KNN "               , KNNModel),
          SKLearnModel("SVM"                , clf),
          SKLearnModel("MLP"                , mlp)]

# Define augmentation parameters
acc_Noise  = 0.001
gyro_Noise = 0.001
augmentation_factor = 5

# Load the data
labels = np.load("TestingData\\labels.npy")
samples = np.load("TestingData\\samples.npy")

# # Let's add the sobel transform of the data as more axis
# new_samples = np.zeros((samples.shape[0], samples.shape[1], samples.shape[2] + 6)) 
# new_samples[:,:, 0:6] = samples

# for dataPoint in new_samples:
    # for i in range(6):
        # dataPoint[:, i + 6] = ndimage.sobel(dataPoint[:,i])

# samples = new_samples

# Determine the number of classes
numClasses = len(np.unique(labels))

# Shuffle the data
newIDX = np.random.permutation(len(samples))
samples = samples[newIDX]
labels = labels[newIDX]

# Smooth the data
samples = MovingAverage(samples, 25)

# Subtract out the mean of the data
for i in range(6):
    samples[:,:,i] = samples[:,:,i] - np.mean(samples[:,:,i])

# Create a list to store the errors
errors = np.zeros((len(models), NumberofFolds))

# Start the progress bar
printProgressBar(0, NumberofFolds, prefix = 'Progress:', suffix = 'Complete', length = 50)

for i in range(NumberofFolds):
    # Flatten the data
    flatSamples = flattenData(samples)
    
    # Split the data into testings and training - 20% split
    # 20% might get us in a case where we don't have one of the classes so instead of taking 20% of the entire space, take 20% of each class and then merge them
    train_img, test_img, train_lbl, test_lbl = SplitData(numClasses, flatSamples, labels, 0.1)
    
    # Create augmented data
    Aug_Samples = np.repeat(train_img, augmentation_factor, axis = 0)
    Aug_Labels  = np.repeat(train_lbl, augmentation_factor, axis = 0)
    Aug_Samples = Aug_Samples.reshape(train_img.shape[0] * augmentation_factor, samples.shape[1], samples.shape[2])
    Aug_Samples[:,:,0:3] = Aug_Samples[:,:,0:3] + np.random.normal(0.0, acc_Noise , size=(Aug_Samples.shape[0], Aug_Samples.shape[1], 3))
    Aug_Samples[:,:,3:6] = Aug_Samples[:,:,3:6] + np.random.normal(0.0, gyro_Noise, size=(Aug_Samples.shape[0], Aug_Samples.shape[1], 3))
    
    # Flatten the augmented data
    flatSamples = flattenData(Aug_Samples)
      
    # Concatenate the training data and the augmented data
    train_img = np.concatenate((train_img, flatSamples), axis=0)
    train_lbl = np.concatenate((train_lbl,  Aug_Labels), axis=0)
    
    # Let's standardize the data
    scaler = MinMaxScaler()
    
    # Fit on training set only.
    scaler.fit(train_img)
    
    # Apply transform to both the training set and the test set.
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img) 
    
    # Create a PCA model with 0.99% variance
    pca = PCA(0.99)
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
        
    # Update the progress bar
    printProgressBar(i + 1, NumberofFolds, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Output all the errors
for j in range(len(models)):
    print(['{:.2f}'.format(ii) for ii in errors[j]], end=" - ")
    print('{:.2f}'.format(np.mean(errors[j])), end=" - ")
    print(models[j].name)

# Confusion Matrices 
if(OutputConfusionMatrix):
    for model in models:
        print(confusion_matrix(test_lbl, model.predict(test_img)))