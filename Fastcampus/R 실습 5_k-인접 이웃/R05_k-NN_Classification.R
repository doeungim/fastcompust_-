# k-Nearest Neigbor Learning (Classification) -----------------------------
# Performance Evaluation Function -----------------------------------------
perf_eval2 <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}

# Initialize the performance matrix
perf_mat <- matrix(0, 2, 6)
colnames(perf_mat) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat) <- c("Logistic Regression", "k-NN")
perf_mat

# kknn package install & call
install.packages("kknn", dependencies = TRUE)
library(kknn)

# Load the wdbc data
RawData <- read.csv("wdbc.csv", header = FALSE)

# Normalize the input data
Class <- as.factor(RawData[,31])
InputData <- RawData[,1:30]
ScaledInputData <- scale(InputData, center = TRUE, scale = TRUE)
ScaledData <- data.frame(ScaledInputData, Class)

# Divide the dataset into the training (70%) and Validation (30%) datasets
set.seed(123)
trn_idx <- sample(1:length(Class), round(0.7*length(Class)))
wdbc_trn <- ScaledData[trn_idx,]
wdbc_tst <- ScaledData[-trn_idx,]

# Classification model 1: Logistic Regression
full_lr <- glm(Class ~ ., family=binomial, wdbc_trn)
summary(full_lr)

lr_response <- predict(full_lr, type = "response", newdata = wdbc_tst)
lr_target <- wdbc_tst$Class
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >= 0.5)] <- 1

cm_logreg <- table(lr_target, lr_predicted)
cm_logreg

perf_mat[1,] <- perf_eval2(cm_logreg)
perf_mat

# Perform k-nn classification with k=3, Distance = Euclidean, and weighted scheme = majority voting
kknn <- kknn(Class ~ ., wdbc_trn, wdbc_tst, k=3, distance=2, kernel = "rectangular")

# View the k-nn results
summary(kknn)
kknn$CL
kknn$W
kknn$D

table(wdbc_tst$Class, kknn$fitted.values)

# Visualize the classification results
knnfit <- fitted(kknn)
table(wdbc_tst$Class, knnfit)
pcol <- as.character(as.numeric(wdbc_tst$Class))
pairs(wdbc_tst[c(1,2,5,6)], pch = pcol, col = c("blue", "red")[(wdbc_tst$Class != knnfit)+1])

# Leave-one-out validation for finding the best k
knntr <- train.kknn(Class ~ ., wdbc_trn, kmax=10, distance=2, kernel="rectangular")

knntr$MISCLASS
knntr$best.parameters

# Perform k-nn classification with the best k, Distance = Euclidean, and weighted scheme = majority voting
kknn_opt <- kknn(Class ~ ., wdbc_trn, wdbc_tst, k=knntr$best.parameters$k, 
                 distance=2, kernel = "rectangular")
fit_opt <- fitted(kknn_opt)
cm_knn <- table(wdbc_tst$Class, fit_opt)
cm_knn

perf_mat[2,] <- perf_eval2(cm_knn)
perf_mat
