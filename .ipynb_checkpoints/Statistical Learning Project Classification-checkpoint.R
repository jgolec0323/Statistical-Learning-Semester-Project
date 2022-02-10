library(ISLR)
library(corrplot)
library(boot)
library(class)
library(MASS)

data = read.csv("Downloads/winequality-red.csv")
wine = data.frame(data)

# correlation plot for attributes
correlation <- cor(wine)
corrplot(correlation, method = "circle")

# histogram of wine quality
hist(wine[,12], main = "Histogram of Wine Quality", xlab = "Rating")

# create wine data for classification
binary = as.numeric(data$quality >= 7)
binary <- factor(binary, labels = c("bad", "good"))
wine_class = cbind(wine[-12], binary)

# split into training and test set
size = floor(0.70*nrow(wine_class))  
set.seed(123)   
train_ind = sample(seq_len(nrow(wine_class)),size = size)  
train = wine_class[train_ind,]
test = wine_class[-train_ind,]

x.train = train[,-12]
y.train = train[,12]
x.test = test[,-12]
y.test = test[,12]

# logistic regression
glm.fit1 <- glm(binary~., data = wine_class, subset = train_ind, family = binomial)
summary(glm.fit1)
set.seed(13)
cv.10.1 = cv.glm(train, glm.fit1, K=10)$delta
cv.10.1

stat_sig = c(2, 5, 7, 10, 11, 12)

# logistic regression with statistically significant predictors
glm.fit2 <- glm(binary ~., data = wine_class[stat_sig], subset = train_ind, family = binomial)
summary(glm.fit2)
cv.10.2 = cv.glm(train, glm.fit2, K=10)$delta
cv.10.2

# predict response and calculate confusion matrix
glm.probs1 = predict(glm.fit1, test, type="response")
glm.pred1 = rep(0, nrow(test))
glm.pred1[glm.probs1 > 0.50] = 1
table(glm.pred1, y.test)
mean(glm.pred1 == y.test)

# predict response and calculate confusion matrix
glm.probs2 = predict(glm.fit2, test, type="response")
glm.pred2 = rep(0, nrow(test))
glm.pred2[glm.probs2 > 0.50] = 1
table(glm.pred2, y.test)
mean(glm.pred2 == y.test)


# LDA
lda.fit = lda(binary ~., data = wine_class, subset = train_ind)
lda.fit
lda.pred = predict(lda.fit, x.test)
table(lda.pred$class, y.test)
mean(lda.pred$class == y.test)

# QDA
qda.fit = qda(binary ~., data = wine_class, subset = train_ind)
qda.fit
qda.pred = predict(qda.fit, x.test)
table(qda.pred$class, y.test)
mean(qda.pred$class == y.test)

# KNN
#-------
standardized.x = scale(wine_class[,-12])
train.x = standardized.x[train_ind,]
test.x = standardized.x[-train_ind,]

set.seed(1)
knn.pred = knn(train.x, test.x, y.train, k = 5)
table(knn.pred, y.test)
mean(knn.pred == y.test)

# SVM
#---------
svm.mod = svm(binary~., data = wine_class, subset = train_ind, kernel = "radial")
summary(svm.mod)
svm.pred = predict(svm.mod, x.test)
table(svm.pred, y.test)
mean(svm.pred == y.test)

# tune using cross validation
set.seed(1)
tune.svm = tune(svm, binary~., data = wine_class[train_ind,],
                ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                gamma = c(0.5,1,2,3,4)))
summary(tune.svm)

best.svm = tune.svm$best.model
summary(best.svm)

svm.pred = predict(best.svm, x.test)
table(svm.pred, y.test)
mean(svm.pred == y.test)


