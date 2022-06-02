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

# Performance table initialization
Perf_table <- matrix(0, nrow = 2, ncol = 3)
colnames(Perf_table)<- c("RMSE", "MAE", 'MAPE')
rownames(Perf_table)<- c("MLR", "Regression Tree")
Perf_table

# Load the dataset
corolla <- read.csv("ToyotaCorolla.csv")

# Regression model 1: multivariate linear regression (MLR)
id_idx <- c(1,2)

# Prepare the data for MLR
corolla_mlr_data <- corolla[,-c(id_idx)]

# Split the data into the training/validation sets
set.seed(123) 
trn_idx <- sample(1:nrow(corolla), round(0.7*nrow(corolla)))

MLR_trn <- corolla_mlr_data[trn_idx,]
MLR_tst <- corolla_mlr_data[-trn_idx,]

# Train the MLR
MLR_corolla <- lm(Price ~ ., data = MLR_trn)

# Performance Measure
MLR_corolla_haty <- predict(MLR_corolla, newdata = MLR_tst)

Perf_table[1,] <- perf_eval_reg(MLR_tst$Price, MLR_corolla_haty)
Perf_table

# Regression model 2: Regression Tree
# Install the necessary package
install.packages("tree")
library(tree)

corolla_rt_data <- corolla[,-id_idx]
RT_trn <- corolla_rt_data[trn_idx,]
RT_tst <- corolla_rt_data[-trn_idx,]

# Training the tree
RT_corolla <- tree(Price ~ ., RT_trn)
summary(RT_corolla)

# Plot the tree
plot(RT_corolla)
text(RT_corolla, pretty = 1)

# Find the best tree
set.seed(123)
RT_corolla_cv <- cv.tree(RT_corolla, FUN = prune.tree)

# Plot the pruning result
plot(RT_corolla_cv$size, RT_corolla_cv$dev, type = "b")
RT_corolla_cv

# Select the final model
RT_corolla_pruned <- prune.tree(RT_corolla, best = 7)
plot(RT_corolla_pruned)
text(RT_corolla_pruned, pretty = 1)

# Prediction
RT_corolla_prey <- predict(RT_corolla_pruned, RT_tst, type = "vector")

# Compare the regression performance
Perf_table[2,] <- perf_eval_reg(RT_tst$Price, RT_corolla_prey)
Perf_table
