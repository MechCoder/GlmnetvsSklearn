from scipy.io import mmwrite, mmread
from sklearn.linear_model.coordinate_descent import _alpha_grid
from sklearn.linear_model import lasso_path
from 
from time import time

# Newsgroup datasets
X_new = mmread("X_new.mtx")
y_new = mmread("y_new.mtx")
y_new = y_new.toarray()[0]
#alphas = _alpha_grid(X_new, y_new, eps=1e-3, fit_intercept=True, normalize=False, n_alphas=100)
t = time()
lasso_path(X_new, y_new, eps=1e-3, precompute=False, fit_intercept=True, normalize=False, n_alphas=100)
print time() - t

X_new = mmread("haxby_X.mtx").toarray()
y_new = mmread("haxby_y.mtx").toarray()[0]
#print y_new.shape
#y_new = y_new.toarray()[0]
# alphas = _alpha_grid(X_new, y_new, eps=1e-3, fit_intercept=True, normalize=False, n_alphas=100)

t = time()
coef = lasso_path(X_new, y_new, eps=1e-3, precompute=False, fit_intercept=True, normalize=False, alphas=alphas)#n_alphas=100)
coef_ = coef[1]
sq_loss = np.sum(0.5*(y_new[:, np.newaxis] - safe_sparse_dot(X_new, coef_))**2, axis=0)
l2_penalty = np.sum(coef_**2, axis=0)
l1_penalty = np.sum(np.abs(coef_), axis=0)

print time() - t