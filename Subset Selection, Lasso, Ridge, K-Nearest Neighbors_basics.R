rm(list=ls(all=TRUE)) 

###########################################################
#### Subset Selection
###########################################################

library(ISLR) #contains the dataset "hitters"
library(leaps) #contains the function regsubsets
head(Hitters) # want to predict Salary based on other variables

#Remove NA entries in Hitters
sum(is.na(Hitters$Salary))
Hitters=na.omit(Hitters) #omits rows corresponding to NA entries
#dim(Hitters)

# Make training and test sets
set.seed(1)
train=sample(1:nrow(Hitters),0.75*nrow(Hitters))
test=-train

#nvmax = maximum number of predictors to consider (default = 8)
#number of predictors in dataset (= 19 in Hitters)
p = ncol(Hitters) - 1; #we subtract the response variable

#Best Subset Selection using Traditional Approach (see Session 7-8)
regfit.full=regsubsets(Salary~.,data=Hitters[train,],nvmax=p)

#regfit.full contains p models, where model t is the best model
#obtained using exactly t predictors (t ranges from 1 to p)
reg.summary=summary(regfit.full)
reg.summary

reg.summary$adjr2
best.model = which.max(reg.summary$adjr2)

predict.regsubsets=function(regfit.full,newdata,t){
  #In this problem, form="Salary~.". It represents the modeling argument we inputted when calling regsubsets()
  form=as.formula(regfit.full$call[[2]])
  mat=model.matrix(form,newdata) #mat = model.matrix(Salary~., newdata)
  coefi=coef(regfit.full,id=t) #obtain the coefficients of the model corresponding to t
  xvars=names(coefi)
  pred = mat[,xvars]%*%coefi
  return(pred)
}

#evaluate the best model on the test set
pred=predict.regsubsets(regfit.full,Hitters[test,],best.model)
actual = Hitters$Salary[test];
mean((actual-pred)^2) #test set MSE

# Forward Stepwise Selection
regfit.fwd=regsubsets(Salary~.,data=Hitters[train,],nvmax=p,method="forward")
best.model.fwd = which.max(summary(regfit.fwd)$adjr2)

coef(regfit.full,best.model)
coef(regfit.fwd,best.model.fwd)

#Best Subset Selection using 10-fold Cross Validation
k=10
set.seed(1)
Hitters.train = Hitters[train,]

folds=sample(1:k,nrow(Hitters.train),replace=TRUE)
#cv.errors[j,t] represents the MSE from the best model using t parameters evaluated on fold j

cv.errors=array(NA,dim=c(k,p)) 
for(j in 1:k){

  best.fit=regsubsets(Salary~.,data=Hitters.train[folds!=j,],nvmax=p)
  #For t=1,...,p, evaluate the best model using t predictors on fold j (Hitters.train[folds==j,])
  for(t in 1:p){
    pred=predict.regsubsets(best.fit,Hitters.train[folds==j,],t)
    actual=Hitters.train$Salary[folds==j]
    #cv.errors[j,t] represents the MSE from the best model using t parameters evaluated on fold j
    cv.errors[j,t]=mean((actual-pred)^2)
  }
}

#average MSEs across the folds j=1,...,k (step 2e)
mean.cv.errors=apply(cv.errors,2,mean)
mean.cv.errors

#compute the "best" number of parameters, t*, through minimizing CV MSEs over t=1,...,p
best.model = which.min(mean.cv.errors)

#find the best model with t* predictors using entire training dataset
regfit.full=regsubsets(Salary~.,data=Hitters.train, nvmax=19)

#evaluate MSE of final chosen model on test dataset
pred=predict.regsubsets(regfit.full,Hitters[test,],best.model)
actual = Hitters$Salary[test];
mean((actual - pred)^2) #test set MSE

###########################################################
#### Ridge and Lasso Regression
###########################################################
#prepare the arguments for glmnet()
x=model.matrix(Salary~.,Hitters)[,-1]
y=Hitters$Salary


library(glmnet)
#Ridge and Lasso regression have a tuneable parameter: lambda (See Session 7-19)
#set sequence of lambdas
grid=10^(-2:10) 

set.seed(65)
#Use 10-fold CV to choose the best value of lambda for ridge regression
cv.out=cv.glmnet(x[train,],y[train],alpha=0,lambda=grid,nfolds=10) 
plot(cv.out)
bestlam=cv.out$lambda.min

#Train model with best value of lambda on the training set
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=bestlam)

#Evaluate this model on the test set
pred=predict(ridge.mod,x[test,])
actual = y[test]
mean((actual-pred)^2) 

##########################################################
##########################################################

# 

# The Stock Market Data

library(ISLR)
names(Smarket)
dim(Smarket)
head(Smarket)
summary(Smarket)
pairs(Smarket)
cor(Smarket[,-9])
attach(Smarket)
plot(Volume)

Smarket %>% select(Year) %>% distinct()

# K-Nearest Neighbors
train=(Year<2005) ##Set train set to be the year before 2005
Smarket.2005=Smarket[!train,]
dim(Smarket.2005)
Direction.2005=Direction[!train]
library(class)
train.X=cbind(Lag1,Lag2)[train,] 
test.X=cbind(Lag1,Lag2)[!train,]
train.Direction=Direction[train]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.2005)
(83+43)/252
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)
mean(knn.pred==Direction.2005)           

