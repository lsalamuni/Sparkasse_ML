###############################################################################
#                                                                             #
#                               1. PACKAGES                                   #
#                                                                             #
###############################################################################

packages = c("e1071", "caret", "ggplot2", "readxl","uwot","dplyr", "MLmetrics")

# Load and install packages if needed 

package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

###############################################################################
#                                                                             #
#                          2. DATA PREPARATION                                #
#                                                                             #
###############################################################################

# Load dataset
library(readxl)
df <- read_xlsx("~/Downloads/ml_dataset_v5_final.xlsx")

# Remove Document ID to prevent data leakage
library(dplyr)
X <- df %>% 
  select(-document)

###############################################################################
#                                                                             #
#                          3. TRAIN/TEST SPLIT                                #
#                                                                             #
###############################################################################

# Split data into training and test data; use seed for reproducility
set.seed(123) 
training_indices <- createDataPartition(X$category, p = 0.8, list = FALSE)
training_data <- X[training_indices, ]
test_data <- X[-training_indices, ]

# Separate features and targets into new dataframes
X_train <- training_data %>% 
  select(-category)
Y_train <- training_data$category

X_test <- test_data %>% 
  select(-category)
Y_test <- test_data$category

###############################################################################
#                                                                             #
#                          4. SCALING DATA                                    #
#                                                                             #
###############################################################################

# SVMs are scale sensitive. Scale train/test data separately to prevent data leakage
# Calculate training data's scaling parameters to use them on test set
# Dataframe contains mixed data including binary features, which results in NAN, due to zero variance. 

# Preserve zero variance columns to avoid NaNs or distortion of binary features with function:
get_scaling_params <- function(m) {
  params <- list()
  # Create a loop to include all columns
  for (col in seq_along(m)) {
    
    # Extract all values, as well as unique values
    vals <- m[[col]]
    uniq_vals <- unique(vals)
    
    # Skip binary columns
    if (length(uniq_vals) == 2 && all(uniq_vals %in% c(0, 1))) {
      next
    }
    
    # Calculate mean and standard deviation
    mean_val <- mean(vals, na.rm = TRUE)
    sd_val <- sd(vals, na.rm = TRUE)
    
    params[[names(m)[col]]] <- list(mean = mean_val, sd = sd_val)
  }
  
  return(params)
}

# Apply scaling for all features except binary, due to zero variance, with this function:
apply_scaling <- function(m, params) {
  
  # Create loop for all columns in params
  for (col in names(params)) {
    
    # Retrieve mean and standard deviation
    mean_val <- params[[col]]$mean
    sd_val <- params[[col]]$sd
    
    # Skip scaling if sd is zero (zero variance)
    if (sd_val == 0) {
      next
    }
    
    # Scale using mean and sd of training data
    m[[col]] <- (m[[col]] - mean_val) / sd_val
  }
  
  return(m)
}

# Now use both functions to scale X_train and X_test with the same scaling parameters of X_train
scaling_params <- get_scaling_params(X_train)

# Training data
X_train_scaled <- apply_scaling(X_train, scaling_params)

# Test data
X_test_scaled <- apply_scaling(X_test, scaling_params)

###############################################################################
#                                                                             #
#                 5. CLASS WEIGHTS AND IMBALANCED DATA                        #
#                                                                             #
###############################################################################

# Check for imbalanced data: Count, percentage per class and visual check
table(Y_train)
prop.table(table(Y_train))
barplot(table(Y_train),
        main = "Class Distribution of Target",
        xlab = "Category",
        ylab = "Count")

# Calculate class weight for imbalanced data and SVM model input
class_counts <- table(Y_train)

# Calculate inverse-frequency weights: large numbers ~ 1 and smaller numbers > 1
class_weights <- max(class_counts)/class_counts
class_weights <- setNames(as.numeric(class_weights), names(class_counts))


###############################################################################
#                                                                             #
#                  6. BASELINE SVM MODEL (LINEAR) & TEST                      #
#                                                                             #
###############################################################################

# Load package for Support Vector Model
library(e1071)

# Convert Y_train and Y_test into factors for SVM classification
Y_train <- as.factor(Y_train)
Y_test <- as.factor(Y_test)

# Train SVM Model
svm_model_benchmark <- svm(x = X_train_scaled, y = Y_train, 
                           kernel = "linear",
                           cost = 1,
                           class.weights = class_weights,
                           scale = FALSE) # Data already scaled in 3rd step

# Test benchmark SVM model by making predictions with test data
predictions <- predict(svm_model_benchmark, newdata = X_test_scaled)

# Create confusion matrix to compare predicted against actual categories for test data
conf_matrix <- table(Predicted = predictions, Actual = Y_test)
print(conf_matrix)

# Calculate accuracy
accuracy <- mean(predictions == Y_test)
cat("Test Accuracy:", accuracy, "\n")

# Also test for macro F1: Treats all classes equally regardless of size (solution for imbalance)

library(MLmetrics)

# Preparations for macro F1: Align same set of possible categories for predictions and actual answers
predictions_F1 <- factor(predictions, levels = levels(Y_test))
Y_test_F1 <- factor(Y_test, levels = levels(Y_test))

# Compute F1 score for each class 1-12
f1_per_class <- sapply(levels(Y_test_F1), function(class) {
  F1_Score(y_true = Y_test_F1, y_pred = predictions_F1, positive = class)
})

# Calculate macro F1
macro_f1 <- mean(f1_per_class)

cat("Macro F1 Score:", macro_f1, "\n")

###############################################################################
#                                                                             #
#                   7. STRATIFIED K-FOLD CROSS-VALIDATION                     #
#                                                                             #
###############################################################################

# For reproducibility
set.seed(111)

# Define the target variable explicitly for cross-validation
target_Y_cv <- X$category

# Create Folds
folds <- createFolds(target_Y_cv, k = 5, list = TRUE, returnTrain = TRUE)

# Define hyperparameters for tuning
kernels <- c("linear", "radial")
cost_values <- c(0.1, 1, 10, 100)
gamma_values <- c(0.001, 0.01, 0.1, 1)

# Create a function for calculating macro F1 manually
calc_macro_f1 <- function(true, pred) {
  pred <- factor(pred, levels = levels(true))
  
  # Loop over each class to do a one vs. rest calculation for each class
  f1s <- sapply(levels(true), function(cl) {
    TP <- sum(true == cl & pred == cl)
    FP <- sum(true != cl & pred == cl)
    FN <- sum(true == cl & pred != cl)
    
    # Handle the case where no predictions or true values for a class exist
    if ((TP + FP) == 0 || (TP + FN) == 0) {
      return(0)
    }
    
    # Compute how many predicted classes were correct (precision) and how many were catched (recall)
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    
    # Avoid division by zero in the F1 calculation itself
    if ((precision + recall) == 0) {
      return(0)
    }
    
    # Harmonic mean to penalize extreme differences between precision and recall
    f1 <- 2 * (precision * recall) / (precision + recall)
    return(f1)
  })
  
  # Ensure the result is not NaN if all f1s are 0
  if (all(is.nan(f1s))) {
    return(0)
  }
  
  # Calculate average F1-score across all classes; any NA values are ignored
  mean(f1s, na.rm = TRUE)
}

# Create emtpy data frame for upcoming results
results_cv <- data.frame()

# Create loop to assign each fold into training and validation dataframes, calculate class weights and perform SVM models for linear and radial kernel models
for (fold_i in seq_along(folds)) {
  train_idx_cv <- folds[[fold_i]]
  val_idx_cv <- setdiff(seq_along(target_Y_cv), train_idx_cv)
  
  # Extract features and target for training and validation sets: exclude category from features and make targets into factors for SVM classification
  X_train_cv <- X[train_idx_cv, setdiff(names(X), "category")]
  Y_train_cv <- as.factor(target_Y_cv[train_idx_cv])
  
  X_val_cv <- X[val_idx_cv, setdiff(names(X), "category")] # Exclude 'category' from features
  Y_val_cv <- as.factor(target_Y_cv[val_idx_cv])
  
  # Calculates class weights within each fold in the loop
  make_weights <- function(y) {
    
    # If one class is absent in training fold it gets no weight
    y <- droplevels(as.factor(y))  
    
    # Same procedure as in 5. Count classes and calculate inverse-frequency weights
    w <- table(y)                        
    w <- max(w) / w              
    
    # Match class weights with names of classes
    setNames(as.numeric(w), names(w))    
  }
  
  w <- make_weights(Y_train_cv)
  
  # Compute scaling parameters of each training data set
  scaling_params_cv <- get_scaling_params(X_train_cv)
  
  # Scale training data
  X_train_cv_scaled <- apply_scaling(X_train_cv, scaling_params_cv)
  
  # Apply same scaling parameters to validation data
  X_val_cv_scaled <- apply_scaling(X_val_cv, scaling_params_cv)
  
  # Create outer and inner loop for testing linear and radial SVM models
  for (kernel in kernels) {
    for (cost in cost_values) {
      
      # Linear kernel (no gamma)
      if (kernel == "linear") {
        
        # Create a SVM model for linear kernel
        svm_model <- svm(x = X_train_cv_scaled, y = Y_train_cv,
                         kernel = kernel,
                         cost = cost,
                         class.weights = w,
                         scale = FALSE)
        
        # Test the SVM model (see above 6.)
        preds <- predict(svm_model, newdata = X_val_cv_scaled)
        
        # Calculate macro F1 (see above 6.)
        macro_f1_val <- calc_macro_f1(Y_val_cv, preds)
        
        # Use 'gamma=NA' for consistency in order to fit the results_cv with results of radial models
        results_cv <- rbind(results_cv, data.frame(kernel=kernel, cost=cost, gamma=NA, fold=fold_i, macroF1=macro_f1_val))
        
        # If gamma is detected, train SVM model with radial kernel
      } else { # radial kernel
        for (gamma in gamma_values) {
          svm_model <- svm(x = X_train_cv_scaled, y = Y_train_cv,
                           kernel = kernel,
                           cost = cost,
                           gamma = gamma,
                           class.weights = w,
                           scale = FALSE)
          preds <- predict(svm_model, newdata = X_val_cv_scaled)
          macro_f1_val <- calc_macro_f1(Y_val_cv, preds)
          results_cv <- rbind(results_cv, data.frame(kernel=kernel, cost=cost, gamma=gamma, fold=fold_i, macroF1=macro_f1_val))
        }
      }
    }
  }
}

# Convert gamma to a factor and replace NA with a new level "linear"
results_cv$gamma <- as.character(results_cv$gamma)
results_cv$gamma[is.na(results_cv$gamma)] <- "linear"
results_cv$gamma <- as.factor(results_cv$gamma)

print("Detailed results for each fold and hyperparameter combination:")
print(results_cv)

# Aggregate results: groups kernel, cost, gamma and computes the mean Macro-F1 across all folds
agg_results <- aggregate(macroF1 ~ kernel + cost + gamma, data = results_cv, FUN = mean)

print("Aggregated Results (Average Macro F1 per parameter combination across all folds):")
print(agg_results)

# Find the best parameters
best_params <- agg_results[which.max(agg_results$macroF1), ]
print("Best Parameters:")
print(best_params)

best_kernel <- as.character(best_params$kernel)
best_cost   <- as.numeric(best_params$cost)

to_num <- function(x) if (is.factor(x)) as.numeric(as.character(x)) else as.numeric(x)

# Find the best gamma only when best_kernel is radial
best_gamma <- if (best_kernel == "radial") to_num(best_params$gamma) else NA_real_

###############################################################################
#                                                                             #
#              8. FINAL SVM WITH BEST PARAMETERS FROM CV + TEST               #
#                                                                             #
###############################################################################
set.seed(123)

# Adjust and train benchmark SVM model
final_svm <- svm(x = X_train_scaled, y = Y_train, 
                 kernel = best_kernel,
                 cost = best_cost,
                 gamma = best_gamma, # best kernel is radial, therefore not an issue
                 class.weights = class_weights,
                 scale = FALSE) # Data already scaled in 3rd step.

final_predictions <- predict(final_svm, newdata = X_test_scaled)
final_conf_matrix <- table(Predicted = final_predictions, Actual = Y_test)
print(final_conf_matrix)

# Calculate accuracy
final_accuracy <- mean(final_predictions == Y_test)
cat("Test Accuracy:", final_accuracy, "\n")

# Calculate macro F1, due to imbalanced classes
final_predictions_F1 <- factor(final_predictions, levels = levels(Y_test))
final_Y_test_F1 <- factor(Y_test, levels = levels(Y_test))

final_macro_f1 <- calc_macro_f1(Y_test, final_predictions)
cat("Macro F1 Score:", final_macro_f1, "\n")

# Compare Macro F1 scores between benchmark and final SVM model
perf_comparison <- data.frame(
  Model = c("Benchmark SVM (Linear)", "Tuned Final SVM"),
  Accuracy = c(accuracy, final_accuracy),
  MacroF1 = c(macro_f1, final_macro_f1)
)
print(perf_comparison)

###############################################################################
#                                                                             #
#                 9. VISUALIZE FINAL SVM MODEL: HEATMAP                       #
#                                                                             #
###############################################################################

# Create a confusion matrix heatmap 
library(dplyr)

# Helper: convert a base::table confusion matrix to tidy df
cm_to_df <- function(cm) {
  df <- as.data.frame(cm)
  colnames(df) <- c("Predicted", "Actual", "Freq")
  df
}

# Normalize by actual class (rows)
normalize_cm <- function(cm_df) {
  cm_df %>%
    group_by(Actual) %>%
    mutate(Prop = Freq / sum(Freq)) %>%
    ungroup()
}

# Generic plotting function
plot_cm_heatmap <- function(cm_df, title, normalized = FALSE) {
  if (normalized) {
    cm_df <- normalize_cm(cm_df)
    ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Prop)) +
      geom_tile(color = "white") +
      geom_text(aes(label = scales::percent(Prop, accuracy = 0.1)), size = 3) +
      scale_fill_gradient(low = "white", high = "darkblue", labels = scales::percent) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        axis.title = element_blank(),
        panel.grid = element_blank()
      ) +
      ggtitle(paste0(title, " (row-normalized)"))
  } else {
    ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile(color = "white") +
      geom_text(aes(label = Freq), size = 4) +
      scale_fill_gradient(low = "white", high = "darkblue") +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        axis.title = element_blank(),
        panel.grid = element_blank()
      ) +
      ggtitle(title)
  }
}

cm_bench_df <- cm_to_df(conf_matrix)
plot_cm_heatmap(cm_bench_df, "Benchmark SVM — Confusion Matrix", normalized = FALSE)
plot_cm_heatmap(cm_bench_df, "Benchmark SVM — Confusion Matrix", normalized = TRUE)

cm_tuned_df <- cm_to_df(final_conf_matrix)
plot_cm_heatmap(cm_tuned_df, "Tuned SVM — Confusion Matrix", normalized = FALSE)
plot_cm_heatmap(cm_tuned_df, "Tuned SVM — Confusion Matrix", normalized = TRUE)

###############################################################################
#                                                                             #
#                     12. FEATURE IMPORTANCE ANALYSIS                         #
#                                                                             #
###############################################################################

simple_perm_importance <- function(model, X, y, metric = calc_macro_f1, nsim = 5) {
  base_score <- metric(y, predict(model, X))
  scores <- sapply(names(X), function(col) {
    drops <- numeric(nsim)
    for (i in seq_len(nsim)) {
      Xp <- X
      Xp[[col]] <- sample(Xp[[col]])  # permute this feature
      pred <- predict(model, Xp)
      drops[i] <- base_score - metric(y, pred)
    }
    mean(drops)  # average drop in score
  })
  sort(scores, decreasing = TRUE)
}

set.seed(42)
imp <- simple_perm_importance(final_svm, X_test_scaled, Y_test, nsim = 10)
head(imp, 20)

library(ggplot2)
imp_df <- data.frame(Feature = names(imp), Importance = as.numeric(imp))
top_n <- 20
imp_df <- imp_df[order(-imp_df$Importance), ][1:top_n, ]

ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 20 Feature Importances (SVM, simple permutation)",
       x = "Feature", y = "Δ Macro F1") +
  theme_minimal(base_size = 12)

