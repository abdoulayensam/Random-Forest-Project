# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caret)

# Generate synthetic data
set.seed(42)
n <- 250
data <- data.frame(
  ID = 1:n,
  Age = sample(10:18, n, replace = TRUE),
  Sexe = sample(c("M", "F"), n, replace = TRUE),
  Pref_Visuel = sample(0:10, n, replace = TRUE),
  Pref_Auditif = sample(0:10, n, replace = TRUE),
  Pref_Kinesthesique = sample(0:10, n, replace = TRUE),
  Math_Score = sample(40:100, n, replace = TRUE),
  Sciences_Score = sample(40:100, n, replace = TRUE),
  Langues_Score = sample(40:100, n, replace = TRUE),
  Temps_Etude_Visuel = runif(n, 1, 10),
  Temps_Etude_Auditif = runif(n, 1, 10),
  Temps_Etude_Kinesthesique = runif(n, 1, 10),
  Satisfaction = sample(1:10, n, replace = TRUE)
)

# Determine learner type
determine_apprenant <- function(row) {
  preferences <- c(
    Visuel = row["Pref_Visuel"],
    Auditif = row["Pref_Auditif"],
    Kinesthesique = row["Pref_Kinesthesique"]
  )
  max_pref <- names(preferences)[which.max(preferences)]
  if (sum(preferences == max(preferences)) > 1) {
    return("Mixte")
  }
  return(max_pref)
}

data$Apprenant_Type <- apply(data, 1, determine_apprenant)

# Convert Apprenant_Type to a factor
data$Apprenant_Type <- as.factor(data$Apprenant_Type)

# Visualizations
par(mfrow = c(3, 4))

# Histograms for scores
hist(data$Math_Score, col = "skyblue", main = "Distribution of Math_Score", xlab = "Math_Score")
hist(data$Sciences_Score, col = "skyblue", main = "Distribution of Sciences_Score", xlab = "Sciences_Score")
hist(data$Langues_Score, col = "skyblue", main = "Distribution of Langues_Score", xlab = "Langues_Score")

# Histograms for study time
hist(data$Temps_Etude_Visuel, col = "skyblue", main = "Distribution of Temps_Etude_Visuel", xlab = "Temps_Etude_Visuel")
hist(data$Temps_Etude_Auditif, col = "skyblue", main = "Distribution of Temps_Etude_Auditif", xlab = "Temps_Etude_Auditif")
hist(data$Temps_Etude_Kinesthesique, col = "skyblue", main = "Distribution of Temps_Etude_Kinesthesique", xlab = "Temps_Etude_Kinesthesique")

# Histogram for satisfaction
hist(data$Satisfaction, col = "skyblue", main = "Distribution of Satisfaction", xlab = "Satisfaction")

# Boxplots for scores by learner type
boxplot(Math_Score ~ Apprenant_Type, data = data, col = "lightgreen", main = "Math Scores by Learner Type", xlab = "Learner Type", ylab = "Math Score")
boxplot(Sciences_Score ~ Apprenant_Type, data = data, col = "lightblue", main = "Science Scores by Learner Type", xlab = "Learner Type", ylab = "Science Score")
boxplot(Langues_Score ~ Apprenant_Type, data = data, col = "pink", main = "Language Scores by Learner Type", xlab = "Learner Type", ylab = "Language Score")

# Implement Decision Tree manually
DecisionTree <- setRefClass(
  "DecisionTree",
  fields = list(max_depth = "numeric", tree = "ANY"),
  methods = list(
    fit = function(X, y) {
      tree <<- .self$build_tree(X, y, depth = 0)
    },
    predict = function(X) {
      return(sapply(1:nrow(X), function(i) .self$predict_row(X[i, ], tree)))
    },
    build_tree = function(X, y, depth) {
      if (length(unique(y)) == 1 || (max_depth > 0 && depth >= max_depth)) {
        return(as.character(names(sort(table(y), decreasing = TRUE)[1])))
      }
      best_feature <- NULL
      best_threshold <- NULL
      best_gain <- -Inf
      for (feature in colnames(X)) {
        if (!is.numeric(X[[feature]])) next  # Skip non-numeric features
        thresholds <- unique(X[[feature]])
        for (threshold in thresholds) {
          gain <- .self$information_gain(X, y, feature, threshold)
          if (is.na(gain)) gain <- 0  # Handle NA values
          if (gain > best_gain) {
            best_gain <- gain
            best_feature <- feature
            best_threshold <- threshold
          }
        }
      }
      if (is.null(best_feature)) {
        return(as.character(names(sort(table(y), decreasing = TRUE)[1])))
      }
      left_indices <- X[[best_feature]] <= best_threshold
      right_indices <- !left_indices
      left_tree <- .self$build_tree(X[left_indices, , drop = FALSE], y[left_indices], depth + 1)
      right_tree <- .self$build_tree(X[right_indices, , drop = FALSE], y[right_indices], depth + 1)
      return(list(feature = best_feature, threshold = best_threshold, left = left_tree, right = right_tree))
    },
    information_gain = function(X, y, feature, threshold) {
      left_y <- y[X[[feature]] <= threshold]
      right_y <- y[X[[feature]] > threshold]
      if (length(left_y) == 0 || length(right_y) == 0) return(0)
      total_entropy <- .self$entropy(y)
      left_entropy <- .self$entropy(left_y)
      right_entropy <- .self$entropy(right_y)
      left_weight <- length(left_y) / length(y)
      right_weight <- length(right_y) / length(y)
      return(total_entropy - (left_weight * left_entropy + right_weight * right_entropy))
    },
    entropy = function(y) {
      counts <- table(y)
      probs <- counts / length(y)
      return(-sum(probs * log2(probs), na.rm = TRUE))
    },
    predict_row = function(row, tree) {
      if (!is.list(tree)) {
        return(tree)
      }
      if (row[[tree$feature]] <= tree$threshold) {
        return(.self$predict_row(row, tree$left))
      } else {
        return(.self$predict_row(row, tree$right))
      }
    }
  )
)

# Define the RandomForest class
RandomForest <- setRefClass(
  "RandomForest",
  fields = list(
    n_trees = "numeric",
    max_depth = "numeric",
    sample_size = "numeric",
    trees = "list"
  ),
  methods = list(
    initialize = function(n_trees = 100, max_depth = 10, sample_size = 0.7) {
      .self$n_trees <- n_trees
      .self$max_depth <- max_depth
      .self$sample_size <- sample_size
      .self$trees <- list()
    },
    fit = function(X, y) {
      for (i in 1:n_trees) {
        sample_indices <- sample(1:nrow(X), size = floor(sample_size * nrow(X)), replace = TRUE)
        X_sample <- X[sample_indices, ]
        y_sample <- y[sample_indices]
        n_features <- floor(sqrt(ncol(X)))
        selected_features <- sample(names(X), n_features)
        X_sample <- X_sample[, selected_features, drop = FALSE]
        tree <- DecisionTree$new(max_depth = max_depth)
        tree$fit(X_sample, y_sample)
        .self$trees[[i]] <- list(tree = tree, features = selected_features)
      }
    },
    predict = function(X) {
      predictions <- matrix(nrow = nrow(X), ncol = n_trees)
      for (i in 1:n_trees) {
        tree <- .self$trees[[i]]$tree
        features <- .self$trees[[i]]$features
        X_selected <- X[, features, drop = FALSE]
        predictions[, i] <- tree$predict(X_selected)
      }
      final_predictions <- apply(predictions, 1, function(row) {
        names(sort(table(row), decreasing = TRUE))[1]
      })
      return(final_predictions)
    }
  )
)

# Train-test split
set.seed(42)
train_indices <- sample(1:nrow(data), 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Prepare data for model training
train_X <- train_data[, !(names(train_data) %in% c("ID", "Apprenant_Type"))]
train_y <- train_data$Apprenant_Type
test_X <- test_data[, !(names(test_data) %in% c("ID", "Apprenant_Type"))]
test_y <- test_data$Apprenant_Type

# Train Decision Tree model
dt_model <- DecisionTree$new(max_depth = 10)
dt_model$fit(train_X, train_y)
y_pred <- dt_model$predict(test_X)
conf_matrix <- table(test_y, y_pred)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Decision Tree accuracy:", accuracy, "\n")

# Train Random Forest model
rf_model <- RandomForest$new(n_trees = 50, max_depth = 10, sample_size = 0.7)
rf_model$fit(train_X, train_y)
rf_predictions <- rf_model$predict(test_X)
rf_conf_matrix <- table(test_y, rf_predictions)
print(rf_conf_matrix)
rf_accuracy <- sum(diag(rf_conf_matrix)) / sum(rf_conf_matrix)
cat("Random Forest accuracy:", rf_accuracy, "\n")
# Plot Decision Tree and Random Forest accuracies
bar_data <- data.frame(
  Model = c("Decision Tree", "Random Forest"),
  Accuracy = c(accuracy, rf_accuracy)
)

ggplot(bar_data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", color = "black") +
  ylim(0, 1) +
  labs(
    title = "Model Accuracies",
    x = "Model",
    y = "Accuracy"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("Decision Tree" = "skyblue", "Random Forest" = "lightgreen")) +
  geom_text(aes(label = round(Accuracy, 2)), vjust = -0.3)
