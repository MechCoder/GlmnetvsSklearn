library(glmnet)
library(Matrix)


# X_new <- readMM("X_new.mtx")
# y_new <- Matrix(readMM("y_new.mtx"))
# ptm <- proc.time()
# fit <- glmnet(X_new, y_new, lambda.min.ratio=0.001, standardize=FALSE, intercept=TRUE)
# proc.time() - ptm
# print(fit)

X_new <- Matrix(readMM("haxby_X.mtx"))
y_new <- Matrix(readMM("haxby_y.mtx"))[1,]
ptm <- proc.time()
glmnet.control(fdev = 0)
fit <- glmnet(X_new, y_new, lambda.min.ratio=0.001, standardize=FALSE,
              intercept=TRUE, maxit=10**7, thresh=1e-7)
fit
proc.time() - ptm

coef_ <- fit$beta
sqloss <- 0.5 * colSums((y_new - (X_new %*% coef_))**2)
l1_penalty <- colSums(abs(c))
l2_penalty <- colSums(c**2) * 0.5
sqloss + l1_penalty + l2_penalty
# t_ = coef(fit, s=0.007202, exact=TRUE)
# which(t_!=0)
# t_[t_ != 0]

# array([[  6.92123804e-17,   2.83210732e-02],
#        [  0.00000000e+00,   5.75088994e-03],
#        [  0.00000000e+00,   2.54628035e-03]])
