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
import Yale_Read_Images
from sklearn.decomposition import PCA
import visualise_yale_file
import random

def crossvalidate(trainlabels,hog_features):
    
    accuracy=np.zeros((5,1))

    
    k=5
    j=0
    '''
    Performs a 5-fold cross-validation using the provided classifier and
    reports performance in terms of precision and recall.
    '''
    

    #skf=StratifiedKFold(trainlabels,k)
    skf = KFold(len(trainlabels), k, shuffle=True,random_state=None)
    for train_index, test_index in skf:
        j=j+1
        X_train, X_test = hog_features[train_index], hog_features[test_index]
        y_train, y_test = trainlabels[train_index],trainlabels[test_index]
       
        
        y_test=np.int32(y_test)
        y_train=np.int32(y_train)
        #Calculate HOG features    
        yalebNB_clf = NB.naivebayes(38)
        yalebNB_clf.fit_classifier(X_train, y_train)
        
        t3 = time()
        yalebNB_clf.predict(X_test)
        print("Cross Validation Testing for fold "+str(j)+" done in %0.3fs" % (time() - t3))

        a=yalebNB_clf.testclass
        
        pred_error = (y_test==a)
        acc=len(np.where(pred_error)[0])/(len(y_test))
        print ("Cross Validation Testing accuracy for fold "+str(j)+" is :" +str(acc*100) +" %")
        accuracy[j-1]=acc
        
    
    meanacc=np.mean(accuracy,axis=0)
#    
    
            
        
    return meanacc,accuracy


random.seed(4)
h = 168
w = 192

#Load complete Extended YaleB dataset from folder CroppedYaleB.
#The images used are the cropped version of the original images. 
#Each image is of size 192 by 168
[X,Y]= Yale_Read_Images.read_images('CroppedYaleB',(h,w))

X1=[]
X2=[]
N=len(X)

size=int(len(X)/38)
A=[X[i:i+size] for i  in range(0, len(X), size)]
L=[Y[i:i+size] for i  in range(0, len(Y), size)]

Xtrain=np.empty((0, h*w))
Xtest=np.empty((0, h*w))
y_train=[]
y_test=[]
#Split the dataset into training and test of equal sizes
# such that each person(class) has 32 images each in training and testing
for i in A:
    random.shuffle(i)
    m=i[0:32]
    n=i[32:64]
    X_train=[]
    X_test=[]
    X1.extend(m)
    X2.extend(n)
    for i in m:
        X_train.append(np.reshape(i,(1,-1)))
    #X_array=np.reshape(np.array(X_train),(32,(h*w)))
    X_array=np.vstack(X_train)
    mean=np.mean(X_array,0)
    Xtrain=np.append(Xtrain,np.abs(X_array-mean),axis=0)
    
    for i in n:
        X_test.append(np.reshape(i,(1,-1)))
    #X_array=np.reshape(np.array(X_test),(32,(h*w)))
    X_array=np.vstack(X_test)

    mean=np.mean(X_array,0)
    Xtest=np.append(Xtest,np.abs(X_array-mean),axis=0)
    
    
random.seed()   
for i in L:    
    y_train.extend(i[0:32])
    y_test.extend(i[32:64])
 

labels=np.unique(y_train)       
#Display Yale B dataset Images for a subset of faces
X_train_show = []
pred_show=[]
X_train_show = X1[0:4] + X1[65:69] + X1[130:134]+X1[195:199]
Y_train_show = y_train[0:4] + y_train[65:69] + y_train[130:134]+y_train[195:199]
visualise_yale_file.visualise_yale(X_train_show,Y_train_show,pred_show,0)

X_test_show = []
Y_test_show=[]
X_test_show = X2[0:4] + X2[65:69] + X2[130:134]+X2[195:199]
Y_test_show = y_test[0:4] + y_test[65:69] + y_test[130:134]+y_test[195:199]



y_train=np.array(y_train)
y_test=np.array(y_test)
 
#Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
#dataset): unsupervised feature extraction / dimensionality reduction
n_components = 300
print("Extracting the top %d eigenfaces from %d faces" %(n_components, Xtrain.shape[0]))
t0 = time()
pca = PCA(n_components = n_components, whiten=True,svd_solver='randomized',random_state=42).fit(Xtrain)
print("done in %0.3fs" % (time() - t0))
eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(Xtrain)
X_test_pca = pca.transform(Xtest)
print("done in %0.3fs" % (time() - t0))



precision=[]
recall = []

#5 FOLD Cross Validation
meanacc,accuracy=crossvalidate(y_train,X_train_pca)    
print("Mean Cross Validation Accuracy is "+str(meanacc[0]*100)+" %")


#Training
t2 = time()
yalebNB_clf = NB.naivebayes(38)
yalebNB_clf.fit_classifier(X_train_pca, y_train)
print("Training done in %0.3fs" % (time() - t2))


#Testing
t3 = time()
yalebNB_clf.predict(X_test_pca)
print("Testing done in %0.3fs" % (time() - t3))
pred=yalebNB_clf.testclass
pred_error = (y_test==pred)
accuracy=len(np.where(pred_error)[0])/(N/2)
print ("Testing accuracy is :" +str(accuracy*100) +" %")

metrics = precision_recall_fscore_support(y_test, pred)
precision=metrics[0]
recall=metrics[1]
 

#Precision and Recall

for i in range(len(labels)):
    print('Precision for class  {} : = {}'  .format(labels[i] ,precision[i]))
    print('Recall for class  {} : = {}'  .format(labels[i] ,recall[i]))
    
    
#View Predictions
pred_show=pred[0:4] + pred[65:69] + pred[130:134]+pred[195:199]

visualise_yale_file.visualise_yale(X_test_show,Y_test_show,pred_show,1)

    
    
# Compute ROC curve and ROC area for each class
n_classes=len(np.unique(y_test))
classes=np.arange(n_classes)
y_test = label_binarize(y_test, classes=classes)
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
for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic YALEB Naive Bayes for first 10 classes ')
plt.legend(loc="lower right")
plt.show()
