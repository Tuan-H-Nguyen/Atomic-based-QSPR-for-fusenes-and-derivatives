"""
Source:
https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.1c00670/suppl_file/ci1c00670_si_001.pdf
from a paper at:
https://doi.org/10.1021/acs.jcim.1c00670
"""
import numpy as np
from numpy import newaxis, concatenate

def padre_features(X1,X2):
    
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    X1 = X1[:,np.newaxis,:].repeat(n2,axis=1)
    X2 = X2[np.newaxis,:,:].repeat(n1,axis=0)

    X_combined = concatenate([X1,X2,X1-X2],axis=2)
    return X_combined.reshape(n1*n2,-1)

def padre_labels(Y1,Y2):
    
    Y1 = Y1[:,newaxis]
    Y2 = Y2[newaxis,:]
    Y_combined = Y1 - Y2
    return Y_combined.flatten()

def padre_train(model, train_X, train_Y):

    X1X2 = padre_features(train_X, train_X)
    print(X1X2.shape)
    Y1_minus_Y2 = padre_labels(train_Y,train_Y)
    print(Y1_minus_Y2.shape)

    model.fit( X1X2, Y1_minus_Y2)

    return model

def padre_predict(model,test_X,train_X,train_Y):
    
    n1 = test_X.shape[0]
    n2 = train_X.shape[0]

    X1X2 = padre_features(test_X,train_X)

    y1_minus_y2_hat = model.predict(X1X2)

    y1_hat_dist = y1_minus_y2_hat.reshape(n1,n2) + train_Y[newaxis,:]

    mean = y1_hat_dist.mean(axis=1)
    std = y1_hat_dist.std(axis=1)

    return mean, std







