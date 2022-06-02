# k-Nearest Neighbor Learning (Regression) --------------------------------
install.packages("FNN", dependencies = TRUE)
library(FNN)

# Performance evaluation function for regression --------------------------
perf_eval_reg <- function(tgt_y, pre_y){
  
  # RMSE
  rmse <- sqrt(mean((tgt_y - pre_y)^2))
  # MAE
  mae <- mean(abs(tgt_y - pre_y))
  # MAPE
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  
  return(c(rmse, mae, mape))
  
}

perf_summary_reg <- matrix(0,2,3)
rownames(perf_summary_reg) <- c("MLR", "k-NN")
colnames(perf_summary_reg) <- c("RMSE", "MAE", "MAPE")

# Concrete strength data
concrete <- read.csv("concrete.csv", header = FALSE)
n_instance <- dim(concrete)[1]
n_var <- dim(concrete)[2]

RegX <- concrete[,-n_var]
RegY <- concrete[,n_var]

# Data Normalization
RegX <- scale(RegX, center = TRUE, scale = TRUE)

# Combine X and Y
RegData <- as.data.frame(cbind(RegX, RegY))

# Split the data into the training/test sets
set.seed(123)
trn_idx <- sample(1:n_instance, round(0.7*n_instance))
trn_data <- RegData[trn_idx,]
tst_data <- RegData[-trn_idx,]

# Regression Model 1: Multiple linear regression
full_model <- lm(RegY ~ ., data = trn_data)
summary(full_model)

mlr_prey <- predict(full_model, newdata = tst_data)
perf_summary_reg[1,] <- perf_eval_reg(tst_data$RegY, mlr_prey)
perf_summary_reg

# Regression Model 2: k-NN regression
# Find the best k using leave-one-out validation
nk <- c(1:10)
n_instance <- dim(trn_data)[1]
n_var <- dim(trn_data)[2]

val_rmse <- matrix(0,length(nk),1)

for (i in 1:length(nk)){
  
  cat("k-NN regression with k:", nk[i], "\n")
  tmp_residual <- matrix(0,n_instance,1)
  
  for (j in 1:n_instance){
    
    # Data separation for leave-one-out validation
    tmptrnX <- trn_data[-j,1:(n_var-1)]
    tmptrnY <- trn_data[-j,n_var]
    tmpvalX <- trn_data[j,1:(n_var-1)]
    tmpvalY <- trn_data[j,n_var]
    
    # Train k-NN & evaluate
    tmp_knn_reg <- knn.reg(tmptrnX, test = tmpvalX, tmptrnY, k=nk[i])
    tmp_residual[j,1] <- tmpvalY - tmp_knn_reg$pred
    
  }
  
  val_rmse[i,1] <- sqrt(mean(tmp_residual^2))
}

# find the best k
val_rmse
best_k <- nk[which.min(val_rmse)]

# Evaluate the k-NN with the test data
test_knn_reg <- knn.reg(trn_data[,1:ncol(trn_data)-1], test = tst_data[,1:ncol(tst_data)-1], 
                        trn_data[,ncol(trn_data)], k=best_k)

tgt_y <- tst_data[,ncol(tst_data)]
knn_haty <- test_knn_reg$pred

perf_summary_reg[2,] <- perf_eval_reg(tgt_y, knn_haty)
perf_summary_reg

# Plot the result
plot(tgt_y, knn_haty, pch = 1, col = "blue", xlim = c(0,80), ylim = c(0, 80))
points(tgt_y, mlr_prey, pch = 2, col = "red", xlim = c(0,80), ylim = c(0,80))
abline(0,1,lty=3)
