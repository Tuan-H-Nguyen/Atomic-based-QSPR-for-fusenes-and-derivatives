import numpy as np

def RMSD(Y,Y_hat):
    """Compute Root mean square deviation

    Args:
        Y (np.array): 
        Y_hat (np.array): 

    Returns:
        float: 
    """
    Y = np.array(Y).reshape(-1)
    Y_hat = np.array(Y_hat).reshape(-1)
    dY = Y_hat - Y
    return np.sqrt((1/len(dY))*np.sum(dY**2))

def compute_r2(x,y):
    """Compute R^2
    Based on equation on wikipedia 
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Args:
        x (np.array): 
        y (np.array): 

    Returns:
        float: R-squared
    """
    sum_xy = np.sum(x*y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_y2 = np.sum(y**2)
    n = len(x)
    r = (n*sum_xy - sum_x*sum_y)/(((n*sum_x2-sum_x**2)*(n*sum_y2-sum_y**2))**0.5)
    return r**2

class RealLoss:
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def denormalize(self,y):
        y = y*self.std + self.mean
        return y

    def __call__(self,y,y_hat):
        y = self.denormalize(y.detach().cpu())
        y_hat = self.denormalize(y_hat.detach().cpu())

        loss = (y - y_hat)**2
        loss = torch.mean(loss,axis = 0)
        
        return loss
