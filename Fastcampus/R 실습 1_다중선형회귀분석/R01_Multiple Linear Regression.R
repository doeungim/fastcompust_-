install.packages("moments")
library(moments)

# Performance evaluation function for regression -------------------------------
perf_eval_reg <- function(tgt_y, pre_y){
  
  # RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  # MAE
  mae <- mean(abs(tgt_y - pre_y))
  # MAPE
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  
  return(c(rmse, mae, mape))
  
}

# Initialize a performance summary
perf_mat <- matrix(0, nrow = 2, ncol = 3)
rownames(perf_mat) <- c("Toyota Corolla", "Boston Housing")
colnames(perf_mat) <- c("RMSE", "MAE", "MAPE")
perf_mat

#-------------------------------------------------------------------------------

# Dataset 1: Toyota Corolla
corolla <- read.csv("ToyotaCorolla.csv")
View(corolla)

# Indices for the inactivated input variables
id_idx <- c(1,2)

# Remove irrelevant columns
corolla_data <- corolla[,-id_idx]

# Check the linearity between X variables and Y variable
# Remove the categorical data (Fuel_Type)
plot_data <- corolla_data[,-6]
corolla_names <- colnames(plot_data)[-1]

par(mfrow = c(5,7))
for (i in 1:length(corolla_names)){
  plot(Price ~ plot_data[,i+1], data = plot_data,
       xlab = corolla_names[i])
}
dev.off()

# One outlier exists for cc variable
cc_outlier <- which(plot_data$cc > 15000)

# Select some linearly related variables with "Price"
plot_data_selected <- plot_data[-cc_outlier,c(1,2,4,5,6,9,13,14)]
corolla_names <- colnames(plot_data_selected)[-1]

par(mfrow = c(2,4))
for (i in 1:length(corolla_names)){
  plot(Price ~ plot_data_selected[,i+1], data = plot_data_selected,
       xlab = corolla_names[i])
}
dev.off()

# Split the data into the training/validation sets
corolla_mlr_data <- corolla_data[-cc_outlier,]
nCar <- nrow(corolla_mlr_data)

# Fix the seed for random number generation
set.seed(12345) 
corolla_trn_idx <- sample(1:nCar, round(0.7*nCar))
corolla_trn_data <- corolla_mlr_data[corolla_trn_idx,]
corolla_tst_data <- corolla_mlr_data[-corolla_trn_idx,]

# Train the MLR
mlr_corolla <- lm(Price ~ ., data = corolla_trn_data)
mlr_corolla
summary(mlr_corolla)
plot(mlr_corolla)

# Plot the result
plot(corolla_trn_data$Price, fitted(mlr_corolla), 
     xlim = c(4000,35000), ylim = c(4000,35000))
abline(0,1,lty=3)

# normality test of residuals
corolla_resid <- resid(mlr_corolla)

m <- mean(corolla_resid)
std <- sqrt(var(corolla_resid))

hist(corolla_resid, density=20, breaks=50, prob=TRUE, 
     xlab="x-variable", main="normal curve over histogram")

curve(dnorm(x, mean=m, sd=std), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

skewness(corolla_resid)
kurtosis(corolla_resid)

# Performance Measure
mlr_corolla_haty <- predict(mlr_corolla, newdata = corolla_tst_data)

perf_mat[1,] <- perf_eval_reg(corolla_tst_data$Price, mlr_corolla_haty)
perf_mat

#-------------------------------------------------------------------------------
# Dataset 2: Boston Housing
boston_housing <- read.csv("BostonHousing.csv")
View(boston_housing)

boston_names <- colnames(boston_housing)[-ncol(boston_housing)]

par(mfrow = c(3,4))
for (i in 1:length(boston_names)){
  plot(MEDV ~ boston_housing[,i], data = boston_housing,
       xlab = boston_names[i])
}
dev.off()

nHome <- nrow(boston_housing)
nVar <- ncol(boston_housing)

# Split the data into the training/validation sets
set.seed(12345)
boston_trn_idx <- sample(1:nHome, round(0.7*nHome))
boston_trn_data <- boston_housing[boston_trn_idx,]
boston_tst_data <- boston_housing[-boston_trn_idx,]

# Train the MLR
mlr_boston <- lm(MEDV ~ ., data = boston_trn_data)
mlr_boston
summary(mlr_boston)
plot(mlr_boston)

# Plot the result
plot(boston_trn_data$MEDV, fitted(mlr_boston), 
     xlim = c(-5,50), ylim = c(-5,50))
abline(0,1,lty=3)

# normality test of residuals
boston_resid <- resid(mlr_boston)

m <- mean(boston_resid)
std <- sqrt(var(boston_resid))

hist(boston_resid, density=20, breaks=50, prob=TRUE, 
     xlab="x-variable", main="normal curve over histogram")

curve(dnorm(x, mean=m, sd=std), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

skewness(boston_resid)
kurtosis(boston_resid)

# Performance Measure
mlr_boston_haty <- predict(mlr_boston, newdata = boston_tst_data)

perf_mat[2,] <- perf_eval_reg(boston_tst_data$MEDV, mlr_boston_haty)
perf_mat
