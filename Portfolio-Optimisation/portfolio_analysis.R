library(tidyverse)
library(caret)
library(glmnet)

# Load and preprocess data
load_and_preprocess_data <- function(file_path) {
  data <- read_csv(file_path)
  X <- data %>% select(-return, -risk, -sharpe_ratio)
  y <- data %>% select(return, risk, sharpe_ratio)
  
  # Split data
  set.seed(42)
  train_index <- createDataPartition(y$return, p = 0.8, list = FALSE)
  X_train <- X[train_index, ]
  X_test <- X[-train_index, ]
  y_train <- y[train_index, ]
  y_test <- y[-train_index, ]
  
  list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
}

# Train elastic net model
train_elastic_net <- function(X_train, y_train) {
  model <- glmnet(as.matrix(X_train), as.matrix(y_train), alpha = 0.5, family = "mgaussian")
  return(model)
}

# Evaluate model
evaluate_model <- function(model, X_test, y_test) {
  predictions <- predict(model, newx = as.matrix(X_test), s = "lambda.min")
  mse <- mean((y_test - predictions)^2)
  r2 <- 1 - sum((y_test - predictions)^2) / sum((y_test - colMeans(y_test))^2)
  list(mse = mse, r2 = r2)
}

# Main function
main <- function() {
  data <- load_and_preprocess_data("portfolio_data.csv")
  
  model <- train_elastic_net(data$X_train, data$y_train)
  
  results <- evaluate_model(model, data$X_test, data$y_test)
  
  cat("Elastic Net Model Results:\n")
  cat("MSE:", results$mse, "\n")
  cat("R2:", results$r2, "\n")
}

main()
