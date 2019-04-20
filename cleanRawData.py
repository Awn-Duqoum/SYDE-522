import json
import os
import shutil
import numpy as np
import scipy.signal
from scipy.interpolate import UnivariateSpline

def onerror(func, path, exc_info):
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise

baseFolder = os.getcwd() + "\\Raw_Data_1"
distFolder = os.getcwd() + "\\TestingData"

def ExtractArray(jsonEntry):
    # Make an empty array
    returnArray = np.zeros((6))
    returnArray[0] = jsonEntry["Accel"]["x"]
    returnArray[1] = jsonEntry["Accel"]["y"]
    returnArray[2] = jsonEntry["Accel"]["z"]
    returnArray[3] = jsonEntry["Gyro"]["x"]
    returnArray[4] = jsonEntry["Gyro"]["y"]
    returnArray[5] = jsonEntry["Gyro"]["z"]
    return returnArray

def ReadJson(jsonData):
    # Create an object to extract the file into
    rawSample = np.zeros((6,len(jsonData["data"])))
    newSample = np.zeros((6,400))
    
    # Extract the data into a useful format
    for ii in range(len(jsonData["data"])):
        rawSample[:,ii] = ExtractArray(jsonData["data"][ii])
    if(len(jsonData["data"]) < 400):
        # Stretch the signal by interpolating it
        new_length = 400
        old_length = len(jsonData["data"])
        old_indices = np.arange(0,old_length)
        new_indices = np.linspace(0,old_length-1,new_length)
        for ii in range(6):
            spl = UnivariateSpline(old_indices,rawSample[ii],k=3,s=0)
    else:
        newSample = rawSample[:,0:400]
        
    return newSample
        
        
# Remove all the data in the dist folder so we don't confuse things
shutil.rmtree(distFolder,  ignore_errors=True, onerror=onerror)
os.mkdir(distFolder)

# Count how many samples we have
sample_count = 0 
for i in range(10):
    sample_count = sample_count + len(os.listdir(baseFolder + "\\" + str(i)))
    
# Create the object to hold the samples
sample = np.zeros((sample_count, 400, 6))
labels = np.zeros((sample_count))

# Create a file counter
fileCounter = 0

for i in range(10):
    # Make the path
    classFolder = baseFolder + "\\" + str(i)
    # Read all the files in the path
    datafiles = os.listdir(classFolder)
        
    for file in datafiles:       
        # Open the file for reading 
        readf  = open(classFolder + "\\" + file, "r")
        # Read the file
        jsonData = json.load(readf)
        # Remove the EOF entry
        a = jsonData["data"].pop() 
        
        # Update the sample object
        sample[fileCounter] = ReadJson(jsonData).T
        
        # Update the label object
        labels[fileCounter] = i

        fileCounter = fileCounter + 1
        
#Save the file
np.save(distFolder + "\\samples", sample)
np.save(distFolder + "\\labels", labels)