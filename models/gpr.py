from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel, RBF
import scipy.optimize

from sklearn.utils.optimize import _check_optimize_result

class ModGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, _max_iter=15000, _gtol = 1e-06 , kernel=None, alpha=1e-10, 
        optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, 
        normalize_y=False, copy_X_train=True, random_state=None):

        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer, 
            n_restarts_optimizer=n_restarts_optimizer,normalize_y=normalize_y, 
            copy_X_train=copy_X_train, random_state=random_state) 

        self._max_iter = _max_iter
        self._gtol = _gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B",
                jac=True, bounds=bounds, options={'maxiter':self._max_iter, 'gtol': self._gtol})
            #_check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
