###############################################################################
#                                                                             #
#                               1. PACKAGES                                   #
#                                                                             #
###############################################################################

packages = c("e1071", "caret", "ggplot2", "readxl","uwot")

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
X <- df[,-1]

###############################################################################
#                                                                             #
#                    3. SPLIT DATASET INTO TRAIN/TEST                         #
#                                                                             #
###############################################################################

set.seed(123) 
training_indices <- createDataPartition(X$category, p = 0.8, list = FALSE)
training_data <- X[training_indices, ]
test_data <- X[-training_indices, ]

###############################################################################
#                                                                             #
#                          4. SCALING DATA                                    #
#                                                                             #
###############################################################################

# SVMs are very scale sensitive 
# Training and test data have to be scaled separately to avoid data leakage
# Separate features and targets for training and test data
X_train <- training_data[,-1]
Y_train <- training_data$category

X_test <- test_data[,-1]
Y_test <- test_data$category

# Calculate scaling parameters of training data in order to use same parameters on test data
# Dataframe contains mixed data including binary features, which results in NAN, due to zero variance. 

# First function: calculates scaling parameters for the whole training data and returns original value, 
# if variance is zero for columns. (Indicating a binary feature, which would otherwise be distorted)

get_scaling_params <- function(m) {
  params <- list()
  
  for (col in seq_along(m)) {
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


# Second function: This scales training and test data with the same parameters
apply_scaling <- function(m, params) {
  for (col in names(params)) {
    mean_val <- params[[col]]$mean
    sd_val <- params[[col]]$sd
    
    # Skip scaling if sd is zero (zero variance)
    if (sd_val == 0) {
      next
    }
    
    # Scale using training mean and sd
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
#                   5. RUN BENCHMARK SVM MODEL - KERNEL: LINEAR               #
#                                                                             #
###############################################################################

library(e1071)

# Convert Y_train and Y_test into factors for SVM
Y_train <- as.factor(Y_train)
Y_test <- as.factor(Y_test)

# Check for imbalanced data: Count, percentage per class and visual check
table(Y_train)
prop.table(table(Y_train))
barplot(table(Y_train),
        main = "Class Distribution of Target",
        xlab = "Category",
        ylab = "Count")

# Calculate class weight for imbalanced data and SVM model input
class_counts <- table(Y_train)
class_weights <- max(class_counts)/class_counts
class_weights <- setNames(as.numeric(class_weights), names(class_counts))

# Train SVM Model
svm_model_benchmark <- svm(x = X_train_scaled, y = Y_train, 
                           kernel = "linear",
                           cost = 1,
                           class.weights = class_weights,
                           scale = FALSE) # Data already scaled in 3rd step.

###############################################################################
#                                                                             #
#                 6. TEST BENCHMARK SVM MODEL - KERNEL:LINEAR                 #
#                                                                             #
###############################################################################

# Test SVM Model
Y_test <- as.factor(Y_test)

predictions <- predict(svm_model_benchmark, newdata = X_test_scaled)
conf_matrix <- table(Predicted = predictions, Actual = Y_test)
print(conf_matrix)

# Calculate accuracy
accuracy <- mean(predictions == Y_test)
cat("Test Accuracy:", accuracy, "\n")

# Calculate macro F1, due to imbalanced classes
predictions_F1 <- factor(predictions, levels = levels(Y_test))
Y_test_F1 <- factor(Y_test, levels = levels(Y_test))

f1_per_class <- sapply(levels(Y_test), function(class) {
  F1_Score(y_true = Y_test, y_pred = predictions_F1, positive = class)
})
macro_f1 <- mean(f1_per_class)

cat("Macro F1 Score:", macro_f1, "\n")

###############################################################################
#                                                                             #
#                        7. STRATIFIED CROSS-VALIDATION                       #
#                                                                             #
###############################################################################

# Choose Kernel, cost and gamma via 5-fold CV
calc_macro_f1 <- function(true, pred) {
  pred <- factor(pred, levels = levels(true))
  
  f1s <- sapply(levels(true), function(cl) {
    TP <- sum(true == cl & pred == cl)
    FP <- sum(true != cl & pred == cl)
    FN <- sum(true == cl & pred != cl)
    
    # Handle the case where no predictions or true values for a class exist
    if ((TP + FP) == 0 || (TP + FN) == 0) {
      return(0)
    }
    
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    
    # Avoid division by zero in the F1 calculation itself
    if ((precision + recall) == 0) {
      return(0)
    }
    
    f1 <- 2 * (precision * recall) / (precision + recall)
    return(f1)
  })
  
  # Ensure the result is not NaN if all f1s are 0
  if (all(is.nan(f1s))) {
    return(0)
  }
  
  mean(f1s, na.rm = TRUE)
}

set.seed(111)

# Define the target variable explicitly
target_Y_cv <- X$category

# Create Folds
folds <- createFolds(target_Y_cv, k = 5, list = TRUE, returnTrain = TRUE)

kernels <- c("linear", "radial")
cost_values <- c(0.1, 1, 10, 100)
gamma_values <- c(0.001, 0.01, 0.1, 1)

results_cv <- data.frame()

# Loop for each fold
for (fold_i in seq_along(folds)) {
  train_idx_cv <- folds[[fold_i]]
  val_idx_cv <- setdiff(seq_along(target_Y_cv), train_idx_cv)
  
  # Extract features and target for training and validation sets
  X_train_cv <- X[train_idx_cv, setdiff(names(X), "category")] # Exclude 'category' from features
  Y_train_cv <- as.factor(target_Y_cv[train_idx_cv])
  
  X_val_cv <- X[val_idx_cv, setdiff(names(X), "category")] # Exclude 'category' from features
  Y_val_cv <- as.factor(target_Y_cv[val_idx_cv])
  
  make_weights <- function(y) {
    y <- droplevels(as.factor(y))       
    w <- table(y)                        
    w <- max(w) / w                      
    setNames(as.numeric(w), names(w))    
  }
  
  w <- make_weights(Y_train_cv)
  
  # Scale training data and get scaling parameters
  scaling_params_cv <- get_scaling_params(X_train_cv)
  X_train_cv_scaled <- apply_scaling(X_train_cv, scaling_params_cv)
  
  # Apply same scaling parameters to validation data
  X_val_cv_scaled <- apply_scaling(X_val_cv, scaling_params_cv)
  
  for (kernel in kernels) {
    for (cost in cost_values) {
      if (kernel == "linear") {
        svm_model <- svm(x = X_train_cv_scaled, y = Y_train_cv,
                         kernel = kernel,
                         cost = cost,
                         class.weights = w,
                         scale = FALSE)
        preds <- predict(svm_model, newdata = X_val_cv_scaled)
        macro_f1_val <- calc_macro_f1(Y_val_cv, preds)
        # Use 'gamma=NA' for consistency
        results_cv <- rbind(results_cv, data.frame(kernel=kernel, cost=cost, gamma=NA, fold=fold_i, macroF1=macro_f1_val))
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

print("Detailed results for each fold and parameter combination:")
print(results_cv)

# Convert gamma to a factor and replace NA with a new level "linear"
results_cv$gamma <- as.character(results_cv$gamma)
results_cv$gamma[is.na(results_cv$gamma)] <- "linear"
results_cv$gamma <- as.factor(results_cv$gamma)

# Aggregate results
agg_results <- aggregate(macroF1 ~ kernel + cost + gamma, data = results_cv, FUN = mean)

print("Aggregated Results (Average Macro F1 per parameter combination):")
print(agg_results)

# Find the best parameters
best_params <- agg_results[which.max(agg_results$macroF1), ]
print("Best Parameters:")
print(best_params)

best_kernel <- as.character(best_params$kernel)
best_cost   <- as.numeric(best_params$cost)

to_num <- function(x) if (is.factor(x)) as.numeric(as.character(x)) else as.numeric(x)

best_gamma <- if (best_kernel == "radial") to_num(best_params$gamma) else NA_real_

###############################################################################
#                                                                             #
#                 8. VISUALIZE RESULT OF CROSS VALIDATION                     #
#                                                                             #
###############################################################################
library(ggplot2)
library(dplyr)

radial_results <- agg_results %>%
  filter(kernel == "radial") %>%
  mutate(cost = as.factor(cost),
         gamma = as.factor(gamma))

ggplot(radial_results, aes(x = cost, y = macroF1, color = gamma, group = gamma)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = "Macro F1-Score vs. Cost for Radial Kernel",
       x = "Cost",
       y = "Average Macro F1-Score",
       color = "Gamma") +
  theme_classic(base_size = 14) +  
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text = element_text(color = "black"),
    axis.title = element_text(face = "bold"),
    legend.position = "right"
  )

###############################################################################
#                                                                             #
#                           9. VISUALIZE DATAFRAME                            #
#                                                                             #
###############################################################################
# Uniform Manifold Approximation and Projection
library(uwot)

UMAP_data <- X_train_scaled

# Run UMAP

set.seed(456) 
umap_result <- umap(UMAP_data, n_neighbors = 15, min_dist = 0.1, metric = "euclidean")

# Create Plot
# Create dataframe with UMAP output
umap_df <- data.frame(
  UMAP1 = umap_result[, 1],
  UMAP2 = umap_result[, 2],
  category = Y_train  
)

# Plot
ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = category)) +
  geom_point(alpha = 0.8, size = 2) +
  theme_minimal() +
  labs(title = "UMAP Projection of training data", x = "UMAP 1", y = "UMAP 2")

###############################################################################
#                                                                             #
#                     10. ADJUSTED SVM MODEL: FINAL                           #
#                                                                             #
###############################################################################
set.seed(123)

# Adjust and train benchmark SVM model
final_svm <- svm(x = X_train_scaled, y = Y_train, 
                           kernel = best_kernel,
                           cost = best_cost,
                           gamma = best_gamma,
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
#                     11. VISUALIZE FINAL SVM MODEL                           #
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