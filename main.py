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
import matplotlib.pyplot as plt

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Flatten Data 
def flattenData(X):
    # Create a structure to fold the data
    flatX = np.zeros((X.shape[0], X.shape[1]*X.shape[2]))
    # For each axis flatten
    for ii in range(X.shape[0]):
        flatX[ii] = X[ii].flatten()
    return flatX 

# Create a structure to more easily test different models
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

# Computes the moving average of a 3-D IMU signal
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
OutputConfusionMatrix = True
errors = []

# Create Models
logisticRegr = LogisticRegression(multi_class="multinomial", max_iter=1e9, solver = 'lbfgs')
KNNModel = KNeighborsClassifier(n_neighbors=5, metric="cosine")
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
mlp = MLPClassifier(hidden_layer_sizes=(64, 128, 256, 512, 512), max_iter=450, alpha=1e-4,
                    solver='adam', tol=1e-8, random_state=1, early_stopping=True,
                    learning_rate_init=0.001, verbose=0)

# Create Model array
models = [SKLearnModel("Logistic Regression", logisticRegr), 
          SKLearnModel("KNN "               , KNNModel),
          SKLearnModel("SVM"                , clf),
          SKLearnModel("MLP"                , mlp)]
          
# Load the data
labels = np.load("TrainingData\\labels.npy")
samples = np.load("TrainingData\\samples.npy")

test_labels = np.load("TestingData2\\labels.npy")
test_samples = np.load("TestingData2\\samples.npy")

# Determine the number of classes
numClasses = len(np.unique(labels))

# Shuffle the training data
newIDX = np.random.permutation(len(samples))
samples = samples[newIDX]
labels = labels[newIDX]

# Shuffle the testing data
newIDX = np.random.permutation(len(test_samples))
test_samples = test_samples[newIDX]
test_labels = test_labels[newIDX]

# Smooth the data
samples = MovingAverage(samples, 25)
test_samples = MovingAverage(test_samples, 25)

# Create a list to store the errors
errors = np.zeros(len(models))

# Flatten the data
train_img = flattenData(samples)
train_lbl = labels

# Create the testing data
test_img = flattenData(test_samples)
test_lbl = test_labels

# Let's standardize the data
scaler = MinMaxScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img  = scaler.transform(test_img) 

# Create a PCA model with 0.95% variance
pca = PCA(0.95)
pca.fit(train_img)

# Transform the data
train_img = pca.transform(train_img)
test_img  = pca.transform(test_img)

# Score each model
for ii in range(len(models)):
    # Train the model
    models[ii].fit(train_img, train_lbl)
    
    # Run the model on test data  
    errors[ii] = models[ii].score(test_img, test_lbl)
    
# Output all the errors
for j in range(len(models)):
    print('{:.2f}'.format(errors[j]), end=" - ")
    print(models[j].name)

# Confusion Matrices 
if(OutputConfusionMatrix):
    np.set_printoptions(precision=2)

    for model in models:
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(test_lbl, model.predict(test_img), classes=class_names, title=model.name + ' Confusion matrix')
        plt.savefig("TestingData2-" + model.name + ' CM', bbox_inches='tight')