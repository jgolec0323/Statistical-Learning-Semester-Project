library(ISLR)
library(corrplot)
library(boot)
library(glmnet)
library(pls)
library(leaps)
library(e1071)

data = read.csv("Downloads/winequality-red.csv")
wine = data.frame(data)

size = floor(0.70*nrow(wine))  
set.seed(123)   
train_ind = sample(seq_len(nrow(wine)),size = size) 

train = wine[train_ind,]
test = wine[-train_ind,]

x.train = train[,-12]
y.train = train[,12]
x.test = test[,-12]
y.test = test[,12]


# fit ordinary linear regression model with all predictors
#---------------------------------------------------------
lm.mod = lm(quality ~., data = wine, subset = train_ind)
summary(lm.mod)
lm.pred = predict.lm(lm.mod, x.test)
mean((lm.pred-y.test)^2)

#create model matrix for best subset, ridge and lasso
x = model.matrix(quality~., wine)[,-1]
y = wine$quality


# fit Best subset selection using k fold cross validation
#---------------------------------------------------------
predict.regsubsets = function(object, newdata, id, ...) {
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}

k = 10
p = ncol(wine) - 1
folds = sample(rep(1:k, length = nrow(wine)))
cv.errors = matrix(NA, k, p)
for (i in 1:k) {
  best.fit = regsubsets(quality ~ ., data = wine[folds != i, ], nvmax = p)
  for (j in 1:p) {
    pred = predict(best.fit, wine[folds == i, ], id = j)
    cv.errors[i, j] = mean((wine$quality[folds == i] - pred)^2)
  }
}
mean.cv.errors = apply(cv.errors, 2, mean)
plot(mean.cv.errors, xlab = "Number of predictors", ylab = "CV RMSE", pch = 19, type = "b")
which.min(mean.cv.errors)
mean.cv.errors[which.min(mean.cv.errors)]

reg.best = regsubsets(quality~., data = wine, subset = train_ind, nvmax = p)
coef(reg.best, 8)
reg.pred = predict(reg.best, test, id=8)
mean((reg.pred-y.test)^2)


# fit Ridge Regression model
#---------------------------
grid = 10^seq(10, -2, length = 100)
ridge.mod = glmnet(x[train_ind,], y[train_ind], alpha=0, lambda=grid)
#ridge.pred = predict(ridge.mod, s=4, newx = x[-train_ind,])
#mean((ridge.predict - y.test)^2)

# cross validation to choose lambda
cv.ridge = cv.glmnet(x[train_ind,], y[train_ind], alpha=0)
plot(cv.ridge)
bestlam = cv.ridge$lambda.min
bestlam
ridge.pred = predict(ridge.mod, s = bestlam, newx = x[-train_ind,])
mean((ridge.pred - y.test)^2)
out = glmnet(x[train_ind,], y[train_ind], alpha=0)
predict(out, type = "coefficients", s = bestlam)[1:12,]



# fit Lasso Regression model
#----------------------------
lasso.mod = glmnet(x[train_ind,], y[train_ind], alpha=1, lambda=grid)
#lasso.predict = predict(lasso.mod, s=4, newx = x[-train_ind,])
#mean((lasso.predict - y.test)^2)

# cross validation to choose lambda
cv.lasso=cv.glmnet(x[train_ind,], y[train_ind], alpha=1)
plot(cv.lasso)
bestlam=cv.lasso$lambda.min
bestlam
lasso.predict = predict(lasso.mod, s = bestlam, newx = x[-train_ind,])
mean((lasso.predict - y.test)^2)
out = glmnet(x[train_ind,], y[train_ind], alpha=1)
predict(out, type = "coefficients", s = bestlam)[1:12,]


# fit Principal Component Regression
#------------------------------------
set.seed(1)
pcr.mod = pcr(quality ~., data = wine, subset = train_ind, scale = TRUE, validation = "CV")
summary(pcr.mod)
validationplot(pcr.mod, val.type = "MSEP")

# test pcr with 'M' number of components 
pcr.pred = predict(pcr.mod, x.test, ncomp = 9)
mean((pcr.pred-y.test)^2)

# fit Partial Least Squares
#---------------------------
set.seed(1)
pls.mod = plsr(quality ~., data = wine, subset = train_ind, scale = TRUE, validation = "CV")
summary(pls.mod)
validationplot(pls.mod, val.type = "MSEP")

pls.pred = predict(pls.mod, x.test, ncomp = 4)
mean((pls.pred-y.test)^2)

pls.mod = plsr(quality ~., data = wine, scale = TRUE, ncomp = 4)
summary(pls.mod)





# fit Support Vector Machine regression
#--------------------------------------
svm.mod = svm(quality~., data = wine, subset = train_ind, scale = TRUE, type = "eps-regression")
summary(svm.mod)
svm.pred = predict(svm.mod, x.test)
mean((svm.pred - y.test)^2)


#tune the svm model with cost (penalty factor for non-seperable points) and epsilon (kernel coefficient for rbf)
# high cost -> may overfit
# low cost -> may undefit

set.seed(1)
tune.svm = tune(svm, quality~., data = wine[train_ind,],
                ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                              epsilon = c(0, 0.01, 0.2)))
summary(tune.svm)
best.svm = tune.svm$best.model
summary(best.svm)
svm.pred = predict(best.svm, x.test)
mean((svm.pred - y.test)^2)
