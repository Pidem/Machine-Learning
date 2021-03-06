
###############Regression Tree
library(MASS)
library(tree)
head(Boston)
#crim per capita crime rate by town.

set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston,pretty=0)  ##add lables to the tree we just plotted
cv.boston=cv.tree(tree.boston)  ##Doing cross validation on the tree
plot(cv.boston$size,cv.boston$dev,type='b')
prune.boston=prune.tree(tree.boston,best=5)  ##Prune the tree with size 5(number of terminal nodes to be 5)
plot(prune.boston)
text(prune.boston,pretty=0)
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

###

# The Stock Market Data

library(ISLR)
names(Smarket)
dim(Smarket)
head(Smarket)
summary(Smarket)
pairs(Smarket)
cor(Smarket)  ##This line won't work since one of variables is not numerical.
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
train.X=cbind(Lag1,Lag2)[train,] ## This means we only use Lag1 and Lag2 in our knn
test.X=cbind(Lag1,Lag2)[!train,]
train.Direction=Direction[train]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction.2005)
(83+43)/252
knn.pred=knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction.2005)
(87+48)/252
mean(knn.pred==Direction.2005)           


head(Caravan)
dim(Caravan)
attach(Caravan)
summary(Purchase)
348/5822
standardized.X=scale(Caravan[,-86])
var(Caravan[,1])
var(Caravan[,2])
var(standardized.X[,1])
var(standardized.X[,2])
test=1:1000
train.X=standardized.X[-test,]
test.X=standardized.X[test,]
train.Y=Purchase[-test]
test.Y=Purchase[test]
set.seed(1)
knn.pred=knn(train.X,test.X,train.Y,k=1)
mean(test.Y!=knn.pred)
mean(test.Y!="No")
table(knn.pred,test.Y)
9/(68+9)
knn.pred=knn(train.X,test.X,train.Y,k=3)
table(knn.pred,test.Y)
5/26
knn.pred=knn(train.X,test.X,train.Y,k=5)
table(knn.pred,test.Y)
4/15

