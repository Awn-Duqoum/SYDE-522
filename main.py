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
        
# Control Output
OutputConfusionMatrix = False
NumberofFolds = 5
errors = []

# Create Models
logisticRegr = LogisticRegression(multi_class="multinomial", max_iter=1e9, solver = 'lbfgs')
KNNModel = KNeighborsClassifier(n_neighbors=1)
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
mlp = MLPClassifier(hidden_layer_sizes=(150,150), max_iter=150, alpha=1e-4,
                    solver='adam', tol=1e-4, random_state=1,
                    learning_rate_init=.01)

# Create Model array
models = [SKLearnModel("Logistic Regression",logisticRegr), 
          SKLearnModel("KNN "               ,KNNModel),
          SKLearnModel("SVM"                ,clf),
          SKLearnModel("MLP"                ,mlp)]

# Load the data
labels = np.load("data\\labels.npy")
samples = np.load("data\\samples.npy")

numClasses = len(np.unique(labels))

# Flatten the data
flatSamples = np.zeros((samples.shape[0], samples.shape[1]*samples.shape[2]))
for i in range(samples.shape[0]):
    flatSamples[i] = samples[i].flatten()

errors = []
for i in range(len(models)): 
    errors.append([])

for i in range(NumberofFolds):
    # Split the data into testings and training - 20% split
    # 20% might get us in a case where we don't have one of the classes so instead of taking 20% of the entire space, take 20% of each class and then merge them
    train_img, test_img, train_lbl, test_lbl = SplitData(numClasses, flatSamples, labels, 0.2)
        
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
        errors[ii].append(models[ii].score(test_img, test_lbl))       

# Output all the errors
for i in range(len(errors)):
    print(['{:.2f}'.format(ii) for ii in errors[i]], end=" - ")
    print('{:.2f}'.format(np.mean(errors[i])), end=" - ")
    print(models[i].name)
