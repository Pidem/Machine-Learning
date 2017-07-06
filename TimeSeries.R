#Analysis of S&P 500

HistPrices<-read.csv("C:/Users/p_mal/Downloads/HistoricalQuotes.csv")
HistPrices$close

SPtimeseries<-ts(rev(HistPrices$close),frequency = 365,start=c(2012,215))
plot.ts(SPtimeseries)

#Smooth Data using a simple moving average
library(TTR)
SP_SMA3<-SMA(SPtimeseries,n=3)
plot.ts(SP_SMA3)

SP_SMA8<-SMA(SPtimeseries,n=8)
plot.ts(SP_SMA8)


#Decompose Seasonality 
SP_components<-decompose(SPtimeseries)
plot(SP_components)

#Forecasting using Triple Exponential Smoothing
SP_forecast<-HoltWinters(SPtimeseries,beta=FALSE,gamma=FALSE)
SP_forecast #alpha coefficient is far away from zero 
SP_forecast$SSE
plot(SP_forecast)

library(forecast)
SP_forecast2<-forecast.HoltWinters(SP_forecast,h=100) #forecasting for the next 100 days
SP_forecast2
plot.forecast(SP_forecast2)

plotForecastErrors <- function(forecasterrors)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors)/4
  mysd <- sd(forecasterrors)
  mymin <- min(forecasterrors) - mysd*5
  mymax <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  #freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

plotForecastErrors(SP_forecast2$residuals)

#ARIMA Models
SPdiff<-diff(SPtimeseries,differences = 1)
plot.ts(SPdiff)
acf(SPdiff,lag.max=30)

SParima<-arima(SPtimeseries,order=c(0,4,1))
SParima
SPforecastarima<-forecast.Arima(SParima,h=100)
SPforecastarima
plot.forecast(SPforecastarima)


