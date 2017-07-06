rm(list = ls()) 

#Cross Validation

#Download Auto.csv 
Auto = read.csv("Auto.csv",na.strings ="?") #replacing all instances of "?" with "NA"
Auto = na.omit(Auto)
head(Auto)
dim(Auto)
attach(Auto)

#Divide the data into a training (70%) and test set (30%)
set.seed(1)
train=sample(1:nrow(Auto),0.7*nrow(Auto))
test = -train 

#Validation set approach: we divide the training data into two sets, a training (70%) and validation set (30%)
#number of elements in our training set
num.train.v = floor(0.7*length(train))
#training set
train.v = train[1:num.train.v] #make train.v first num.train.v elements (we can do this because train is a random sample)
#validation set
valid.v = train[(num.train.v+1):length(train)]

#fit a polynomial model
fit01= lm(mpg~poly(horsepower,2),data=Auto[train.v,])
summary(fit01)

#train model on training set and estimate MSE on validation set
mse = rep(0,5)
for (i in 1:5){
  lm.fit=lm(mpg~poly(horsepower,i),data=Auto[train.v,])
  mpg.pred = predict(lm.fit,newdata = Auto[valid.v,])
  mse[i] = mean((mpg[valid.v]-mpg.pred)^2)
}
best.model = which.min(mse)

#train best model on training + validation set
lm.fit=lm(mpg~poly(horsepower,best.model),data=Auto[train,])
#evaluate model on test set
mpg.pred = predict(lm.fit,newdata = Auto[test,])
mse.best.model = mean((mpg[test]-mpg.pred)^2)

#Cross-Validation

library(boot)

cv.error=rep(0,5) #cross validation MSEs for our 5 models
for (i in 1:5){
  glm.fit=glm(mpg~poly(horsepower,i),data=Auto[train,])
  #delta[1] corresponds to MSE for object "cv.glm(Auto,glm.fit)"
  #cv.error[i]=cv.glm(Auto[train,],glm.fit)$delta[1] # Leave-One-Out Cross-Validation
  cv.error[i]=cv.glm(Auto[train,],glm.fit,K=10)$delta[1] #K-fold cross validation (K=10)
}
best.model = which.min(cv.error)

#train best model on entire training set
lm.fit=lm(mpg~poly(horsepower,best.model),data=Auto[train,])
#evaluate model on test set
mpg.pred = predict(lm.fit,newdata = Auto[test,])
mse.best.model = mean((mpg[test]-mpg.pred)^2)

detach(Auto)

#Bootstrapping
library(ISLR) #contains the dataset "Portfolio"
head(Portfolio) #contains two possibly correlated stocks, X and Y
#The minimum variance portfolio is the value of alpha which minimizes Var(alpha*X + (1-alpha)*Y)


#outputs an estimate for alpha using the data in data[indices,], 
alpha.fn = function(data,indices){
  X = data$X[indices]
  Y = data$Y[indices]
  alpha = (var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)) #our estimate of alpha* on this sample
  return(alpha)
}

B = 1000 #number of bootstrap samples
boot(Portfolio,alpha.fn,B) 

res = rep(0,B) #our estimate of alpha* for each bootstrap sample
for (i in 1:B){
  indices = sample(1:nrow(Portfolio),nrow(Portfolio),replace=T)
  res[i] = alpha.fn(Portfolio,indices)
}
hist(res) 
mean(res) 
sd(res)