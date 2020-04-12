import random
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(seed=10)

def get_train_test_split(X, Y):
    
    N = X.shape[0]
    selection = np.random.choice([True, True, True, True, False], N, replace=True)

    X_train = X[selection,:]
    y_train = Y[selection]
    X_test = X[np.invert(selection), :]
    y_test = Y[np.invert(selection)]
    
    return X_train, y_train, X_test, y_test
    
    
def get_color(i):

    if i < 10:
        return '#eb4034'
    elif i<40:
        return '#eb9634'
    elif i<100:
        return '#aeeb34'
    elif i<200:
        return '#34eb3a'
    elif i<400:
        return '#34c3eb'
    elif i<600:
        return '#344ceb'
    elif i<800:
        return '#9f34eb'
    elif i<950:
        return '#521d4e'
    else:
        return 'black'
        
def get_XY(mode, X, Y, i):
    
    if mode == 'all':
        
        return X, Y
    
    elif mode == 'batch':
        
        N = X.shape[0]
        
        batch_size = 32
        
        sample = np.random.randint(0,N, (1,int(batch_size)))

    elif mode == 'sgd':
        
        N = X.shape[0]
      
        sample = np.random.randint(0,N, (1,1))
    
    sample = sample[0]
    
    X = X[sample,:]  
    
    Y = Y[sample]   

    return X, Y

def get_plot(X, Y, X_test, y_test, train_error, test_error, weights, mode):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

    ax[0].scatter(X[:,1].T, Y, alpha=0.6, s=3, c='black')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    
    for i, w in enumerate(weights):

        Y_hat = np.dot(X_test, w)
        
        ax[0].plot(X_test[:,1], Y_hat, c=get_color(i), alpha=0.4, linewidth=0.5)
    
    ax[1].plot(train_error, linewidth=1, label='train_MSE', c='red')
    ax[1].plot(test_error, linewidth=1, label='test_MSE', c='blue')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MSE (' + mode + ')' )
    
    print("Lowest MSE: ", np.min(test_error))
    plt.legend()
    plt.tight_layout()
    plt.show()