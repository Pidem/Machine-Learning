rm(list = ls()) 

#Linear Regression
Ad = read.csv("Advertising.csv")

#Seperate dataset to training set and test set
# trainRows = runif(nrow(Ad))>0.25
# train = Ad[trainRows,]
# test = Ad[!trainRows,]

#method 2
set.seed(1)
train = sample(1:nrow(Ad),0.75*nrow(Ad))
test = -train

#fit two linear regression models
fit1 = lm(Sales~Radio+Newspaper,data=train)
summary(fit1)

fit2 = lm(Sales~TV+Radio,data=train)
summary(fit2)

attach(Ad)

fit1 = lm(Sales~TV+Newspaper,data=Ad[train,]) 
fit2 = lm(Sales~TV+Radio,data=Ad[train,])   

#test models on test dataset
Sales.test.pred1 = predict(fit1,newdata = Ad[test,])
Sales.test.pred2 = predict(fit2,newdata = Ad[test,])

mse1= mean((Sales[test] - Sales.test.pred1)^2)
mse2= mean((Sales[test] - Sales.test.pred2)^2)

#mse2 is much lower. seems that the second model is a better representative of the data.

#####
#Generate an interaction term.
Ad$TV_Newspaper = TV*Newspaper
fit3 = lm(Sales~TV+Newspaper+TV_Newspaper,data=Ad[train,])
summary(fit3)

fit4 = lm(Sales~TV*Newspaper,data=Ad[train,])
summary(fit4)

#Confidence intervals of the sample coefficients on Tv and Newspaper (default 0.95)
confint(fit4)

#Prediction interval
predict(fit4, data.frame(TV = 60, Newspaper = 1300), interval = "prediction", level = 0.95)
predict(fit4, data.frame(TV = 60, Newspaper = 1300), interval = "prediction", level = 0.99)


#Regression with non linear independent variables
attach(Ad)
l1 = lm(Sales~TV)
summary(l1)

l2 = lm(Sales~I(TV^2)) #regresses on the square of TV
summary(l2)

#######################################
#Creating visualizations
library(dplyr)
require(Hmisc)

#Load data
eCarData = read.csv("e-Car_Data--Extract.csv")  

#######
#Generate bins for APR
eCarData$RateBin = cut2(eCarData$Rate, c(4,4.5,5, 5.5, 6, 6.5,7, 7.5, 8, 8.5))

#Select Outcomes and the bins for APR, group by the APR bins, and calculate the average for each bin. 
Conversion_Rate_VS_APR= eCarData %>% select(Outcome, RateBin) %>% group_by(RateBin) %>% summarise(Conversion_Rate = mean(Outcome))

#generate bar chart
barplot(Conversion_Rate_VS_APR$Conversion_Rate, ylab = "Conversion Rate", xlab= "APR", main = "Conversion Rate vs APR", names.arg = Conversion_Rate_VS_APR$RateBin)
barplot(Conversion_Rate_VS_APR$Conversion_Rate, ylab = "Conversion Rate", xlab= "APR", main = "Conversion Rate vs APR")

#######
# Generate a variable call count, and sum it up after grouping by APR bins.
Quotes_by_APR = eCarData %>% mutate(count = 1) %>% select(count, RateBin) %>% group_by(RateBin) %>% summarise(Quotes = sum(count)) 

# Generate bar chart
barplot(Quotes_by_APR$Quotes, ylab = "Number of Quotes", xlab = "APR", main = "Number of Quotes by APR", names.arg = Quotes_by_APR$RateBin)

college_private = college %>% filter(Private == 1)


