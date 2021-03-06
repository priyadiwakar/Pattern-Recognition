
from __future__ import division
import os
import struct
import numpy as np
from skimage.feature import hog
from time import time
import classifierNB as NB
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle




def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image,label,pred):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    s="True Label : "+str(label)+" Predicted label : "+str(pred)
    pyplot.xlabel(s,fontname="Arial", fontsize=20 )
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
def crossvalidate(trainlabels,hog_features):
    
    accuracy=np.zeros((5,1))
    
    
    k=5
    j=0
    '''
    Performs a 5-fold cross-validation using the provided classifier and
    reports performance in terms of precision and recall.
    '''
    
    #skf=StratifiedKFold(trainlabels,k)
    skf = KFold(len(trainlabels), k, shuffle=False,random_state=None)
    for train_index, test_index in skf:
        j=j+1
        X_train, X_test = hog_features[train_index], hog_features[test_index]
        y_train, y_test = trainlabels[train_index],trainlabels[test_index]
    
        
        
        y_test=np.int32(y_test)
        y_train=np.int32(y_train)
        #Calculate HOG features    
        digitsNB_clf = NB.naivebayes(10)
        digitsNB_clf.fit_classifier(X_train, y_train)
        
        
        t3 = time()
        digitsNB_clf.predict(X_test)
        print("Cross Validation Testing for fold "+str(j)+" done in %0.3fs" % (time() - t3))
        a=digitsNB_clf.testclass
        
        pred_error = (y_test==a)
        acc=len(np.where(pred_error)[0])/(len(y_test))
        print ("Cross Validation Testing accuracy for fold "+str(j)+" is :" +str(acc*100) +" %")
        accuracy[j-1]=acc
        
    
    meanacc=np.mean(accuracy,axis=0)
    
    
            
        
    return meanacc,accuracy


#load MNIST database from files train-images.idx3-ubyte,train-labels.idx1-ubyte
#t10k-images.idx3-ubyte,t10k-labels.idx1-ubyte
    
#the MNIST database available online is split into training and test but of unequal sizes
#so after loading the complete dataset we split it into equal halves of size 35000 each
trainingdata=list(read(dataset = "training", path = "."))

N=len(trainingdata)

features=np.zeros((N,784))
labels=np.zeros(N)
list_hog_fd=[]

testingdata=list(read(dataset = "testing", path = "."))

N1=len(testingdata)
features1=np.zeros((N1,784))
labels1=np.zeros(N1)
list_hog_fd1=[]


for i in range(N):
    
    label,pixels=trainingdata[i]
    a=np.reshape(pixels,(1,784))
    features[i,:]=a
    labels[i]=label

for i in range(N1):
    
    label1,pixels1=testingdata[i]
    a=np.reshape(pixels1,(1,784))
    features1[i,:]=a
    labels1[i]=label1
    
#the training and test datasets    
trainingset=features[0:35000,:]
testingset=np.append(features[35000:60000,:],features1,axis=0)
trainlabels=labels[0:35000]
testlabels=np.append(labels[35000:60000],labels1)


#Calculate HOG features    
t0 = time()
for feature in trainingset:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print("HOG features for training set extracted in %0.3fs" % (time() - t0))



t1 = time()
for feature in testingset:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),visualise=False)
    list_hog_fd1.append(fd)
hog_features1 = np.array(list_hog_fd1, 'float64')
print("HOG features for testing set extracted in %0.3fs" % (time() - t1))

precision=[]
recall = []

#5 FOLD Cross Validation
meanacc,accuracy=crossvalidate(trainlabels,hog_features)    

print("Mean Cross Validation Accuracy is "+str(meanacc[0]*100)+" %")

#Training

t2 = time()
digitsNB_clf = NB.naivebayes(10)
digitsNB_clf.fit_classifier(hog_features, trainlabels)
print("Training done in %0.3fs" % (time() - t2))


#Testing
t3 = time()
digitsNB_clf.predict(hog_features1)
print("Testing done in %0.3fs" % (time() - t3))
pred=digitsNB_clf.testclass
pred_error = (testlabels==pred)
accuracy=len(np.where(pred_error)[0])/((N+N1)/2)
print ("Testing accuracy is :" +str(accuracy*100) +" %")


metrics = precision_recall_fscore_support(testlabels, pred)
precision=metrics[0]
recall=metrics[1]

 
labels=np.unique(testlabels) 


#Precision and Recall
for i in range(len(labels)):
    print('Precision for class  {} : = {}'  .format(labels[i] ,precision[i]))
    print('Recall for class  {} : = {}'  .format(labels[i] ,recall[i]))

#View Predictions
j=1
idx=np.argwhere(pred_error==False)
idx1=np.argwhere(pred_error==True)
for i in range(12,43,8):
    k=idx[i][0]
    l=idx1[i][0]
    
    a=np.reshape(testingset[k,:],(28,28))
    show(a,testlabels[k],pred[k])
    a=np.reshape(testingset[l,:],(28,28))
    show(a,testlabels[l],pred[l])
    


# Compute ROC curve and ROC area for each class
n_classes=len(np.unique(testlabels))
classes=np.arange(n_classes)
y_test = label_binarize(testlabels, classes=classes)
y_score = label_binarize(pred, classes=classes)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
lw=2
# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic MNIST Naive Bayes')
plt.legend(loc="lower right")
plt.show()

