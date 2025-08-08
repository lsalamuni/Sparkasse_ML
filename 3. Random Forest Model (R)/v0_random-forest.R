pacotes <- c("readxl", "dplyr", "randomForest", "caret", "ggplot2", "MLmetrics")

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){ 
  instalador <- pacotes[!pacotes %in% installed.packages()] 
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T)
} else {
  sapply(pacotes, require, character = T)
}

#PREPARING DATA FOR MANIPULATION

mail_data <- read_xlsx("ml_dataset_v5_final.xlsx")
#Convert category to a factor
mail_data$category <- as.factor(mail_data$category)

#Remove ID
rf_data <- mail_data %>% select(-document)


# Reweight the classes because of imbalance
class_counts <- table(rf_data$category)
weights <- 1 / class_counts
weights <- weights / sum(weights)



# Fit the first forest as a baseline model
set.seed(123)
rf_model <- randomForest(
  category ~ .,
  data = rf_data,
  ntree = 100,
  classwt = as.list(weights)
)
print(rf_model)

varImpPlot(rf_model)



#TUNING


# Split the data set 60/40 for training and testing, stratified ---
set.seed(123)
train_idx <- createDataPartition(rf_data$category, p = 0.6, list = FALSE)
saveRDS(train_idx, "train_idx.rds")

train_data <- rf_data[train_idx, ]
test_data  <- rf_data[-train_idx, ]

# Save CV folds for seeds 1 to 10 (only run once) 
seeds <- 1:10
folds_list <- list()

for (s in seeds) {
  set.seed(s)
  folds_list[[as.character(s)]] <- createFolds(train_data$category, k = 5)
}
saveRDS(folds_list, "cv_folds_by_seed_10.rds")

# Load saved folds (used in the tuning loop below)
folds_list <- readRDS("cv_folds_by_seed_10.rds")

#Hyperparameter tuning over ntree and seeds ---
ntree_grid <- c(50, 100, 200, 300, 500, 1000)
all_results <- data.frame()

for (s in seeds) {
  folds <- folds_list[[as.character(s)]]
  
  for (ntree_val in ntree_grid) {
    f1_scores <- c()
    log_losses <- c()
    
    for (fold in folds) {
      rf_val   <- train_data[fold, ]
      rf_train <- train_data[-fold, ]
      
      rf_model_cv <- randomForest(
        category ~ ., 
        data = rf_train,
        ntree = ntree_val,
        mtry = floor(sqrt(ncol(rf_train) - 1)), 
        nodesize = 5,
        classwt = as.list(weights)
      )
      
      preds <- predict(rf_model_cv, rf_val)
      probs <- predict(rf_model_cv, rf_val, type = "prob")
      
      cm <- confusionMatrix(preds, rf_val$category)
      f1_scores <- c(f1_scores, mean(cm$byClass[, "F1"], na.rm = TRUE))
      
      y_val_onehot <- model.matrix(~ rf_val$category - 1)
      log_losses <- c(log_losses, MultiLogLoss(y_true = y_val_onehot, y_pred = probs))
    }
    
    all_results <- rbind(
      all_results,
      data.frame(
        Seed = s,
        ntree = ntree_val,
        Macro_F1 = mean(f1_scores),
        LogLoss  = mean(log_losses)
      )
    )
  }
}

#Aggregate results over seeds ---
summary_ntree <- all_results %>%
  group_by(ntree) %>%
  summarise(
    Macro_F1_Mean = mean(Macro_F1),
    Macro_F1_SD   = sd(Macro_F1),
    LogLoss_Mean  = mean(LogLoss),
    LogLoss_SD    = sd(LogLoss),
    .groups = "drop"
  ) %>%
  arrange(desc(Macro_F1_Mean))

print(summary_ntree)



#Second stage of cross validation: 

#Define the seed list and best ntree value from previous tuning
seeds <- 1:10
ntree_best <- 500
mtry_grid <- c(9, 16, 24, 40)
nodesize_grid <- c(5, 7, 10)

# Load saved folds (created beforehand with fixed seeds)
folds_list <- readRDS("cv_folds_by_seed_10.rds")

# Initialize result storage
all_results <- data.frame()

# Loop over seeds and hyperparameters
for (s in seeds) {
  folds <- folds_list[[as.character(s)]]
  
  for (m in mtry_grid) {
    for (node in nodesize_grid) {
      
      f1_scores <- c()
      log_losses <- c()
      
      for (fold in folds) {
        rf_val   <- train_data[fold, ]
        rf_train <- train_data[-fold, ]
        
        # Set seed before training to fix RF randomness
        set.seed(s)
        
        rf_model_cv <- randomForest(
          category ~ ., 
          data = rf_train,
          ntree = ntree_best,
          mtry = m,
          nodesize = node,
          classwt = as.list(weights)
        )
        
        preds <- predict(rf_model_cv, rf_val)
        probs <- predict(rf_model_cv, rf_val, type = "prob")
        
        cm <- confusionMatrix(preds, rf_val$category)
        f1_scores <- c(f1_scores, mean(cm$byClass[, "F1"], na.rm = TRUE))
        
        y_val_onehot <- model.matrix(~ rf_val$category - 1)
        log_losses <- c(log_losses, MultiLogLoss(y_true = y_val_onehot, y_pred = probs))
      }
      
      all_results <- rbind(
        all_results,
        data.frame(
          Seed = s,
          mtry = m,
          nodesize = node,
          Macro_F1 = mean(f1_scores),
          LogLoss  = mean(log_losses)
        )
      )
    }
  }
}

# Aggregate over seeds
summary_results <- all_results %>%
  group_by(mtry, nodesize) %>%
  summarise(
    Macro_F1_Mean = mean(Macro_F1),
    Macro_F1_SD   = sd(Macro_F1),
    LogLoss_Mean  = mean(LogLoss),
    LogLoss_SD    = sd(LogLoss),
    .groups = "drop"
  ) %>%
  arrange(desc(Macro_F1_Mean))

print(summary_results)



#Final model on the train data set

# Fix seed and folds for reproducability
set.seed(123)  # Ensures same fold split
folds <- createFolds(rf_data$category, k = 5, list = TRUE, returnTrain = TRUE)

#Final hyperparameters
final_mtry <- 40
final_nodesize <- 7
ntree_best <- 500

# Store for fold-wise metrics 
f1_scores_macro <- c()
f1_scores_weighted <- c()
log_losses <- c()

all_preds <- factor()
all_true  <- factor()
all_probs <- matrix(nrow = 0, ncol = length(levels(rf_data$category)))

# Set same seed inside loop for internal RF randomness ---
for (fold in folds) {
  rf_train <- rf_data[fold, ]
  rf_val   <- rf_data[-fold, ]
  
  set.seed(123)  # Fix randomness inside RF training
  rf_model_cv <- randomForest(
    category ~ ., 
    data = rf_train,
    ntree = ntree_best,
    mtry = final_mtry,
    nodesize = final_nodesize,
    classwt = as.list(weights)
  )
  
  preds <- predict(rf_model_cv, rf_val)
  probs <- predict(rf_model_cv, rf_val, type = "prob")
  
  cm <- confusionMatrix(preds, rf_val$category)
  
  f1_scores_macro <- c(f1_scores_macro, mean(cm$byClass[, "F1"], na.rm = TRUE))
  
  class_support <- as.numeric(table(rf_val$category))
  f1_scores_weighted <- c(
    f1_scores_weighted,
    weighted.mean(cm$byClass[, "F1"], class_support, na.rm = TRUE)
  )
  
  y_val_onehot <- model.matrix(~ rf_val$category - 1)
  log_losses <- c(log_losses, MultiLogLoss(y_true = y_val_onehot, y_pred = probs))
  
  all_preds <- c(all_preds, preds)
  all_true  <- c(all_true, rf_val$category)
  all_probs <- rbind(all_probs, probs)
}

# Fold-wise mean ± SD metrics
cat(sprintf("Macro-F1: %.3f ± %.3f\n", mean(f1_scores_macro), sd(f1_scores_macro)))
cat(sprintf("Weighted-F1: %.3f ± %.3f\n", mean(f1_scores_weighted), sd(f1_scores_weighted)))
cat(sprintf("Log Loss: %.3f ± %.3f\n", mean(log_losses), sd(log_losses)))

# Aggregated Confusion Matrix
cv_cm <- confusionMatrix(all_preds, all_true)
print(cv_cm$table)
print(cv_cm$byClass[,"F1"])
mean(cv_cm$byClass[,"F1"], na.rm = TRUE)

#Aggregated Log Loss
y_all_onehot <- model.matrix(~ all_true - 1)
agg_logloss <- MultiLogLoss(y_true = y_all_onehot, y_pred = all_probs)
agg_logloss


#Heat map
# Convert confusion matrix to data frame
cm_df <- as.data.frame(cv_cm$table)
colnames(cm_df) <- c("Prediction", "Reference", "Freq")

ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 3) +
  scale_fill_gradient(low = "white", high = "darkblue") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y = element_text(size = 8),
    axis.title = element_blank(),
    panel.grid = element_blank()
  ) +
  ggtitle("Aggregated Confusion Matrix")



#Evaluation on Test Set

# Train final Random Forest on full training data
set.seed(123)
final_rf <- randomForest(
  category ~ ., 
  data = train_data,        # only training data
  ntree = 500,      
  mtry = 40,        
  nodesize = 7,
  classwt = as.list(weights)
)

#Predictions on test set ---
test_preds <- predict(final_rf, test_data)
test_probs <- predict(final_rf, test_data, type = "prob")

#Confusion matrix
cm_test <- confusionMatrix(test_preds, test_data$category)
print(cm_test$table)            # Confusion matrix
print(cm_test$byClass[, "F1"])  # Per-class F1

# Macro-F1 and Weighted-F1
macro_f1_test <- mean(cm_test$byClass[, "F1"], na.rm = TRUE)

# Weighted-F1 = weighted mean of per-class F1 by support
class_support_test <- as.numeric(table(test_data$category))
weighted_f1_test <- weighted.mean(cm_test$byClass[, "F1"], class_support_test, na.rm = TRUE)

cat(sprintf("Test Macro-F1: %.3f\n", macro_f1_test))
cat(sprintf("Test Weighted-F1: %.3f\n", weighted_f1_test))

# Multi-class Log Loss
y_test_onehot <- model.matrix(~ test_data$category - 1)
log_loss_test <- MultiLogLoss(y_true = y_test_onehot, y_pred = test_probs)
cat(sprintf("Test Log Loss: %.3f\n", log_loss_test))

#Train-Test Gap 
# Compare with mean CV Macro-F1
train_macro_f1_cv <- mean(f1_scores_macro)  # from CV
train_test_gap <- macro_f1_test - train_macro_f1_cv
cat(sprintf("Train-Test Macro-F1 Gap: %.3f\n", train_test_gap))



#For comparison with the baseline model

#Baseline Random Forest with default parameters (except for class weights)
set.seed(123)
baseline_rf <- randomForest(
  category ~ .,
  data = train_data,
  ntree = 100,  # default value
  classwt = as.list(weights)
)

#Predict on test set
baseline_preds <- predict(baseline_rf, test_data)
baseline_probs <- predict(baseline_rf, test_data, type = "prob")

# Evaluate metrics
baseline_cm <- confusionMatrix(baseline_preds, test_data$category)

baseline_macro_f1 <- mean(baseline_cm$byClass[, "F1"], na.rm = TRUE)
baseline_weighted_f1 <- weighted.mean(
  baseline_cm$byClass[, "F1"],
  as.numeric(table(test_data$category)),
  na.rm = TRUE
)

baseline_log_loss <- MultiLogLoss(
  y_true = model.matrix(~ test_data$category - 1),
  y_pred = baseline_probs
)

# Print comparison
cat("Baseline Macro-F1: ", round(baseline_macro_f1, 3), "\n")
cat("Baseline Weighted-F1: ", round(baseline_weighted_f1, 3), "\n")
cat("Baseline Log Loss: ", round(baseline_log_loss, 3), "\n")




#FINAL FIT ON ALL DATA (feature importance)

set.seed(123)
#  on full dataset
final_rf <- randomForest(
  category ~ .,
  data = rf_data,       #full dataset now
  ntree = 500,   
  mtry = 40,
  nodesize = 7,  
  classwt = as.list(weights)
)

print(final_rf)


# Feature importance plot
varImpPlot(final_rf)


#For readability
# Extract importance as a dataframe
imp <- data.frame(Feature = rownames(importance(final_rf)),
                  Importance = importance(final_rf)[, 1])  # MeanDecreaseGini

# Sort by importance
imp <- imp[order(imp$Importance, decreasing = TRUE), ]

# Take top N for readability, e.g., top 20
top_n <- 20
imp_top <- head(imp, top_n)

# Plot
ggplot(imp_top, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Feature Importances (Random Forest)",
       x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal(base_size = 12)
