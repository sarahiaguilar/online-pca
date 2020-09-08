library(keras)
library(tensorflow)
library(tfestimators)
library(factoextra)
library(randomForest)
library(caret)
library(data.table)
library(ggplot2)

# Import MNIST dataset
mnist <- keras::dataset_mnist()

## -- Train data -- ##
X <- mnist$train$x
y <- mnist$train$y

# Transform X matrix to vector (28 * 28 = 784)
X <- keras::array_reshape(X, c(nrow(X), 784))

##### Batch PCA (traditional PCA)
batch_pca <- prcomp(X)

##### Online PCA
k <- 50 # Maximum number of principal componentes
n0 <- 100 # Number of samples we'll initialize algorithm with

# Calculates batch PCA with a small sample to initialize algorithm
online_pca <- prcomp(X[1:n0, ])

# Variables to update in algorithm
xbar <- online_pca$center # Average for centering (column means)
online_pca <- list(values = online_pca$sdev[1:k]^2, # Eigenvalues 
                   vectors = online_pca$rotation[, 1:k]) # Eigenvectors (Rotation matrix)

n <- dim(X)[1] # Sample size
for (i in (n0 + 1):n){
  # Update average
  xbar <- onlinePCA::updateMean(xbar, # Last average
                                X[i, ], # Sample we'll use to update PCA
                                i - 1) # Sample size before updating
  
  # Update PCA
  online_pca  <- onlinePCA::incRpca(lambda = online_pca$values, # Eigenvalues
                                    U = online_pca$vectors, # Eigenvectors
                                    x = X[i, ], # Sample we'll use to update PCA
                                    n = i - 1, # Sample size before upsating
                                    q = k, # Number of principal components to calculate
                                    center = xbar) # Average
}

# Function that centers dataset 
center_apply <- function(x) {
  apply(x, 2, function(y) y - mean(y))
}

X_centered <- center_apply(X)
online_pca$x <- as.matrix(X_centered) %*% as.matrix(online_pca$vectors) 


##### Comparison using 50 principal components
sum((batch_pca$sdev[1:50])^2) / sum((batch_pca$sdev)^2) # 82.46%
sum(online_pca$values[1:50]) / sum((batch_pca$sdev)^2) # 82.26%


## -- Test data -- ##
X_test <- mnist$test$x
y_test <- mnist$test$y

# Transform X matrix to vector (28 * 28 = 784)
X_test <- keras::array_reshape(X_test, c(nrow(X_test), 784))

##### Batch PCA (traditional PCA)
batch_pca_test <- predict(batch_pca, X_test)

##### Online PCA
X_test_centered <- center_apply(X_test)
online_pca_test <- as.matrix(X_test_centered) %*% as.matrix(online_pca$vectors) 


##### Visualizations to choose k
variance <- batch_pca$sdev[1:50]^2
proportions_variance <- variance/sum(variance)

# Proportion of explained variance 
plot(proportions_variance, xlab = 'Principal component', ylab = 'Proportion of explained variance', ylim = c(0, 0.15), type = 'b')

# Cumulative proportion of explained variance 
plot(cumsum(proportions_variance), xlab = 'Principal component', ylab = 'Cumulative proportion of explained variance', ylim = c(0, 1), type = 'b')


##### Random forest classifier
set.seed(42)
random_sample_train <- sample(1:60000, 10000)
random_sample_test <- sample(1:10000, 10000)


### All variables 
X_sampled <- X[random_sample_train, ]
y_sampled <- y[random_sample_train]

# Train model 
rf_all <- randomForest(X_sampled, factor(y_sampled), maxnodes = 10)

# Predict y from test data
y_pred_all <- predict(rf_all, X_test)

# Confusion matrix
confusionMatrix(factor(y_pred_all), factor(y_test))


### 50 principal components from batch PCA
X_batch_sampled <- batch_pca$x[random_sample_train, 1:50]

# Train model 
rf_batch_50 <- randomForest(X_batch_sampled, factor(y_sampled), maxnodes = 10)

# Predict y from test data
y_pred_batch_50 <- predict(rf_batch_50, batch_pca_test)

# Confusion matrix
confusionMatrix(factor(y_pred_batch_50), factor(y_test))


### 50 principal components from online PCA
X_online_sampled <- online_pca$x[random_sample_train, 1:50]

# Train model 
rf_online_50 <- randomForest(X_online_sampled, factor(y_sampled), maxnodes = 10)

# Predict y from test data
y_pred_online_50 <- predict(rf_online_50, online_pca_test)

# Confusion matrix
confusionMatrix(factor(y_pred_online_50), factor(y_test))


##### Visualizing first two components
# Batch PCA
ggplot(as.data.table(X_batch_sampled), aes(x = PC1, y = PC2, color = factor(y_sampled))) + 
  geom_point() +
  stat_ellipse(type = "norm")

# Online PCA
ggplot(as.data.table(X_online_sampled), aes(x = V1, y = V2, color = factor(y_sampled))) + 
  geom_point() +
  stat_ellipse(type = "norm")

