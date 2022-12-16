
# coding: utf-8

# In[5]:


import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    # class_1_data = []
    # sum_1 = 0
    # IMPLEMENT THIS METHOD
#     for i,j in zip(X,y):
# #         print(i)
#         for num in range(1,6):
#             if(j == num):
#                 print(f"This is class {num}")
#                 print(i)
#             class_1_data.append(i[0])
    means_x1 = []
    means_x2 = []
    for num in range(1,6):
        class_data_1 = []
        class_data_2 = []
        for i,j in zip(X,y):
            if(j == num):
                class_data_1.append(i[0])
                class_data_2.append(i[1])
#         print(class_data_1)
        means_x1.append(np.mean(class_data_1))
        means_x2.append(np.mean(class_data_2))
    
    
#         print(means_x1)
#         print(means_x2)
        
    means = np.array([means_x1,means_x2])
#     print(means)
    covmat = np.cov(X.T)
#     print(covmat)
    #     summ = np.mean(class_1_data)
#     print(summ)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    means_x1 = []
    means_x2 = []
    for num in range(1,6):
        class_data_1 = []
        class_data_2 = []
        for i,j in zip(X,y):
            if(j == num):
                class_data_1.append(i[0])
                class_data_2.append(i[1])
#         print(class_data_1)
        means_x1.append(np.mean(class_data_1))
        means_x2.append(np.mean(class_data_2))
    
    
#         print(means_x1)
#         print(means_x2)
        
    means = np.array([means_x1,means_x2])
#     print(means)
    
    #---------------------------------- Calculate covariance matrices for each class -----------------------
    covmats = []
    for num in range(1,6):
        class_data = []
        for i,j in zip(X,y):
            if(j == num):
                class_data.append(i)
        class_data_np = np.array(class_data)   
        covmat_class = np.cov(class_data_np.T)
        covmats.append(covmat_class)
        
    covmats = np.array(covmats)
#     print(covmats[0])
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    new_means = means.T
#     print(new_means)
    predicted_values = []

    # IMPLEMENT THIS METHOD
    for i in Xtest:
        
        predicted_prob = []
        for j in range(5):
            inv_cov_mat = np.linalg.inv(covmat)
            mean_each_class = new_means[j]
            x_sub_mu = np.subtract(i,mean_each_class)
            x_sub_mu_transpose = x_sub_mu.T
            multiplication = np.matmul(x_sub_mu_transpose,inv_cov_mat)
            md = np.matmul(multiplication,x_sub_mu)
            md_final = md * (-0.5)
            exp_md = np.exp(md_final)
            c_t_denominator = ((2*np.pi)**2) * ((np.linalg.det(covmat))**0.5)
            predicted_prob.append((1/c_t_denominator)*exp_md)
#             subtraction = np.subtract(i,new_means[j])
#             calc_1 = np.dot(subtraction.T,np.linalg.inv(covmat))
#             subtraction_2 = np.subtract(i,new_means[j])
#             probability = np.dot(calc_1,subtraction_2)
#             probability_res = float(probability)* (-0.5)
#             e_raised_to_probability = np.exp(probability_res)
#             constant_term =  ((2*np.pi)**2) * (np.linalg.det(covmat)) ** 0.5
#             final_result = np.exp(probability)
#             predicted_prob.append(probability)
#             predicted_prob.append((1/constant_term)*e_raised_to_probability) 
#         print(predicted_prob)
        predicted_values.append(predicted_prob.index(max(predicted_prob))+1)
#     print(predicted_values)
    
    
#     print(ytest)
    count = 0
    for pred,true in zip(predicted_values,ytest):
        
        if(pred == true):
            count = count + 1
    
    
    acc = count/len(ytest) * 100
#     print(acc)
    ypred = predicted_values
    ypred = np.array(ypred)
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    new_means = means.T
#     print(new_means)
    predicted_values = []
    # IMPLEMENT THIS METHOD
    for i in Xtest:
        
        predicted_prob = []
        for j in range(5):
            inv_cov_mat = np.linalg.inv(covmats[j])
            mean_each_class = new_means[j]
            x_sub_mu = np.subtract(i,mean_each_class)
            x_sub_mu_transpose = x_sub_mu.T
            multiplication = np.matmul(x_sub_mu_transpose,inv_cov_mat)
            md = np.matmul(multiplication,x_sub_mu)
            md_final = md * (-0.5)
            exp_md = np.exp(md_final)
            c_t_denominator = ((2*np.pi)) * ((np.linalg.det(covmats[j]))**0.5)
            predicted_prob.append((1/c_t_denominator)*exp_md)
#         print(predicted_prob)
        predicted_values.append(predicted_prob.index(max(predicted_prob))+1)
#     print(predicted_values)
    
    
#     print(ytest)
    count = 0
    for pred,true in zip(predicted_values,ytest):
        
        if(pred == true[0]):
            count = count + 1
    
    
    acc = count/len(ytest) * 100
#     print(acc)
    ypred = predicted_values
    ypred = np.array(ypred)
    
    return acc,ypred


def learnOLERegression(X,y):
    w = np.dot((np.linalg.inv(np.dot(X.T,X))),np.dot(X.T,y))
    return w
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD                                                   
#     return w

def learnRidgeRegression(X,y,lambd):
#     shape = X[1].shape
    
#     shape=X.shape[1]
#     print(shape)
    identity_mat = np.identity(X.shape[1])
#     identity_mat = np.identity(shape)
#     print(identity_mat)
    nd_part = np.dot(lambd,identity_mat)
#     print(nd_part)
    w1_1 = np.dot(X.T,X)
    w1_2 = np.dot(lambd,identity_mat)
    w_1 = w1_1 + w1_2
    w_1_inv = np.linalg.inv(w_1)
    w2 = np.dot(X.T,y)
    w = np.dot(w_1_inv,w2)
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    summation = 0
    
    for i, j in zip(Xtest,ytest):
        result = (j - np.dot(w.T,i)) ** 2
        summation = summation + result
    mse = summation / len(ytest)
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):
    w_mat = np.transpose(np.mat(w))
    w_mat_t = np.transpose(w_mat)
#     print(w_mat_t)
    p_1 = np.matmul(X,w_mat)
#     #print(p_1.shape)
    f_p = np.subtract(y,p_1)
#     #print(f_p)
    first_term = np.dot(f_p.T,f_p)
#     print(first_term.shape)
#     print("First Term:")
#     print(first_term)
    w_t_w = np.matmul(w_mat_t,w_mat)
    second_term = lambd * w_t_w
    #print("second")
    #print(second_term)
    #print(second_term.shape)
    error = (0.5 * first_term) + (0.5 * second_term)
    #print(error)
    
    grad_p_1 = np.matmul(y.T,X)
    grad_p_1_t = grad_p_1.T
    grad_p_2 = np.matmul(np.matmul(X.T,X),w_mat)
    grad_p_3 = lambd * w_mat
    error_grad = np.ndarray.flatten(np.array(np.add(np.subtract(grad_p_2,grad_p_1_t), grad_p_3)))
#     print(error_grad.shape)

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
#     # Inputs:                                                                  
#     # x - a single column vector (N x 1)                                       
#     # p - integer (>= 0)                                                       
#     # Outputs:                                                                 
#     # Xp - (N x (p+1)) 
	
#     # IMPLEMENT THIS METHOD
#     x_np = np.array(x)
#     print(x_np.T)
    Xp_init = np.ones((len(x),p+1))
    for row in range(len(x)):
        for col in range(p+1):
            Xp_init[row][col] = x[row] ** col
#     print(Xp_init)
    
    Xp = np.array(Xp_init)
        
    return Xp

# Main script
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
# print(ldares)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# # plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

min_mses3 = np.ndarray.tolist(mses3)
min_index = min_mses3.index(min(min_mses3))

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
#     print(w_d1)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

