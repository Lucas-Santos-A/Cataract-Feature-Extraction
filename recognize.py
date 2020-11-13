# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import SVC
from imutils import paths
import argparse
import cv2
import os
import regex as re
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score
import pickle
from matplotlib import pyplot

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-t", "--training", required=True,
#	help="path to the training images")
#ap.add_argument("-e", "--testing", required=True,
#	help="path to the tesitng images")
#args = vars(ap.parse_args())
# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 3) #Era 24,3
data = []
labels = []
y_test = []
y_pred = []
y_pred_number = []
y_pred_proba = []
# loop over the training images
for imagePath in paths.list_images(os.getcwd()+"/images/training"):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split(os.path.sep)[-2])
	print(imagePath.split(os.path.sep)[-1])
	data.append(hist)
# train a Linear SVM on the data
for label in labels:
        print(label)

model = SVC(C=200, random_state=42, probability=True) #C era 200 / random_state era 42
model.fit(data, labels)

# loop over the testing images
for imagePath in paths.list_images(os.getcwd()+"/images/testing"):
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	#img = cv2.resize(image, (800, 800))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist.reshape(1, -1))
	predictict_proba = model.predict_proba(hist.reshape(1, -1))

	#if prediction[0] == 'cataract':
	#	predNumber = 1
	#else:
	#	predNumber = 0

	if prediction == 'cataract':
		y_pred_number.append(1)
	else:
		y_pred_number.append(0)

	y_pred.append(prediction)
	y_pred_proba.append(predictict_proba)

	if re.search('\Acataract', imagePath.split(os.path.sep)[-1], re.IGNORECASE):
		y_test.append(1)
	else:
		y_test.append(0)

	print(imagePath.split(os.path.sep)[-1])
	#print(predNumber)


filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

	# display the image and the prediction
	#cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
	#	1.0, (0, 0, 255), 3)
	#cv2.imshow("Image", image)
	#cv2.waitKey(0)

print("y_pred")
print(y_pred)

print("y_pred_proba")
print(y_pred_proba)

#ns_probs = [0 for _ in range(len(y_test))]

#ns_auc = roc_auc_score(y_test, ns_probs)
#lr_auc = roc_auc_score(y_test, y_pred_proba)

#print('No Skill: ROC AUC=%.3f' % (ns_auc))
#print('Logistic: ROC AUC=%.3f' % (lr_auc))

#ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
#lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_proba)

#pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

#pyplot.xlabel('False Positive Rate')
#pyplot.ylabel('True Positive Rate')

#pyplot.legend()

#pyplot.show()

#pyplot.savefig("grafico")

print("AUC Score = "+str(roc_auc_score(y_test, y_pred_number)))
print(confusion_matrix(y_test, y_pred_number))
print(classification_report(y_test, y_pred_number))

