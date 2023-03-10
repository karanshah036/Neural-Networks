# -*- coding: utf-8 -*-
"""Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gtJ3-uRWWQgFHwSJTi-i_4LcNGa-KMQm
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    train_data = np.append(train_data, np.ones((train_data.shape[0], 1)), 1)
    
    
    w = initialWeights
#     print()
#     print(w.shape)
#     theta = np.dot(w,train_data)
#     print(train_data.shape)
    theta = sigmoid(np.dot(train_data,w.T))
    theta = np.reshape(theta,(n_data,1))
#     print(theta.shape)
#     log_theta = np.transpose(np.log(theta))
#     print(log_theta[0][0])
    log_theta = np.log(theta)
    first_part = np.dot(log_theta.T,labeli)
#     print(first_part)
    one_minus_y = np.subtract(1,labeli)
    log_one_minus_theta = np.log(np.subtract(1,theta))
    second_part = np.dot(one_minus_y.T,log_one_minus_theta)
    
    err_temp = first_part + second_part 
    error = (-1)*(err_temp/n_data)
#     print(error)
    
#     print("theta",theta.shape)
#     print("labels",labeli.shape)
    err_grad_first_part = np.subtract(theta,labeli)
#     print(err_grad_first_part)
    err_grad_temp = np.dot(err_grad_first_part.T,train_data)
    error_grad_temp = (err_grad_temp/n_data)
    error_grad = error_grad_temp
    error_grad = np.ndarray.flatten(error_grad)
    # print(f"{labeli} -----> {error}")
    return error, error_grad

def blrPredict(W, data):
#     """
#      blrObjFunction predicts the label of data given the data and parameter W 
#      of Logistic Regression
     
#      Input:
#          W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
#          vector of a Logistic Regression classifier.
#          X: the data matrix of size N x D
         
#      Output: 
#          label: vector of size N x 1 representing the predicted label of 
#          corresponding feature vector given in data matrix

#     """
    label = np.zeros((data.shape[0], 1))

#     ##################
#     # YOUR CODE HERE #
#     ##################
#     # HINT: Do not forget to add the bias term to your input data
    
    data = np.append(data,np.ones((data.shape[0],1)),1)
    posterior_prob = np.dot(data,W)
    predicted_val = sigmoid(posterior_prob)
    prediction = np.argmax(predicted_val, axis=1)
    label = np.reshape(prediction,(data.shape[0],1))
    # print(label)
    
    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data


    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label

"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
random_indices = np.random.choice(train_data.shape[0],10000)


random_train_data = train_data[random_indices,:]
random_train_label = train_label[random_indices,:]



print("This is linear SVM implementation")
clf = svm.SVC(kernel='linear')
clf.fit(random_train_data,np.ravel(random_train_label))
pred_labels_train = clf.predict(random_train_data)

print('\n Training set Accuracy:' + str(100 * np.mean((pred_labels_train == np.ravel(random_train_label)).astype(float))) + '%')

pred_labels_validation = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((pred_labels_validation == np.ravel(validation_label)).astype(float))) + '%')

pred_labels_test = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((pred_labels_test == np.ravel(test_label)).astype(float))) + '%')

print("This is default RBF kernel SVM implementation")

clf = svm.SVC(kernel='rbf')
clf.fit(random_train_data,np.ravel(random_train_label))
pred_labels_train = clf.predict(random_train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((pred_labels_train == np.ravel(random_train_label)).astype(float))) + '%')

pred_labels_validation = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((pred_labels_validation == np.ravel(validation_label)).astype(float))) + '%')

pred_labels_test = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((pred_labels_test == np.ravel(test_label)).astype(float))) + '%')

print("This is RBF kernel SVM implementation with gamma=1")

clf = svm.SVC(kernel='rbf', gamma=1)
clf.fit(random_train_data,np.ravel(random_train_label))
pred_labels_train = clf.predict(random_train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((pred_labels_train == np.ravel(random_train_label)).astype(float))) + '%')

pred_labels_validation = clf.predict(validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((pred_labels_validation == np.ravel(validation_label)).astype(float))) + '%')

pred_labels_test = clf.predict(test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((pred_labels_test == np.ravel(test_label)).astype(float))) + '%')

print("This is RBF SVM implementation with gamma=default and C values ranging from 1 to 100")
C_values = [1,10,20,30,40,50,60,70,80,90,100]
train_accuracies = []
validation_accuracies = []
test_accuracies = []
for i in C_values:

  print(f"C------------------------->{i}")
  c_val = i
  clf = svm.SVC(kernel='rbf', C=c_val)
  clf.fit(random_train_data,np.ravel(random_train_label))
  pred_labels_train = clf.predict(random_train_data)
  print('\n Training set Accuracy:' + str(100 * np.mean((pred_labels_train == np.ravel(random_train_label)).astype(float))) + '%')
  train_accuracy = 100 * np.mean((pred_labels_train == np.ravel(random_train_label)).astype(float))
  train_accuracies.append(train_accuracy)

  pred_labels_validation = clf.predict(validation_data)
  print('\n Validation set Accuracy:' + str(100 * np.mean((pred_labels_validation == np.ravel(validation_label)).astype(float))) + '%')
  validation_accuracy = 100 * np.mean((pred_labels_validation == np.ravel(validation_label)).astype(float))
  validation_accuracies.append(validation_accuracy)

  pred_labels_test = clf.predict(test_data)
  print('\n Testing set Accuracy:' + str(100 * np.mean((pred_labels_test == np.ravel(test_label)).astype(float))) + '%')
  test_accuracy = 100 * np.mean((pred_labels_test == np.ravel(test_label)).astype(float))
  test_accuracies.append(test_accuracy)

xpoints = np.array(train_accuracies)
ypoints = np.array(C_values)

print(xpoints.shape)
print(ypoints.shape)
plt.title("C vs train accuracies")
plt.xlabel("C values")
plt.ylabel("Train accuracies")
plt.plot(ypoints, xpoints)
plt.show()

xpoints = np.array(validation_accuracies)
ypoints = np.array(C_values)

plt.title("C vs validation accuracies")
plt.xlabel("C values")
plt.ylabel("Validation accuracies")
plt.plot(ypoints, xpoints)
plt.show()

xpoints = np.array(test_accuracies)
ypoints = np.array(C_values)

plt.title("C vs test accuracies")
plt.xlabel("C values")
plt.ylabel("Test accuracies")
plt.plot(ypoints, xpoints)
plt.show()

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

