# Breast cancer diagnosis using Logistic Regression in R
data <- read.csv("breast_cancer.csv")
data$diagnosis <- factor(data$diagnosis, levels = c("benign", "malignant"))
print(table(data$diagnosis))

# Train (80%) and test (20%) split
n <- nrow(data)
train_size <- floor(0.8 * n)
train <- data[1:train_size, ]
test  <- data[(train_size + 1):n, ]

# Build the logistic regression model
model <- glm(diagnosis ~ radius + texture + smoothness + concavity,
             data = train, family = binomial)
print(summary(model))

# Predict the probability of malignant and classify
prob <- predict(model, newdata = test, type = "response")
pred <- factor(ifelse(prob > 0.5, "malignant", "benign"),
               levels = c("benign", "malignant"))

# Confusion matrix and metrics
cm <- table(Predicted = pred, Actual = test$diagnosis)
print(cm)
accuracy    <- sum(diag(cm)) / sum(cm)
sensitivity <- cm["malignant", "malignant"] / sum(cm[, "malignant"])
specificity <- cm["benign", "benign"] / sum(cm[, "benign"])
cat("Accuracy    :", round(accuracy, 4), "\n")
cat("Sensitivity :", round(sensitivity, 4), "\n")
cat("Specificity :", round(specificity, 4), "\n")

# ---------- Charts ----------
barplot(table(data$diagnosis), col = c("steelblue", "tomato"),
        main = "Benign vs Malignant Cases", ylab = "Number of patients")

boxplot(radius ~ diagnosis, data = data, col = c("steelblue", "tomato"),
        main = "Mean Radius by Diagnosis", ylab = "Mean radius")

# ROC curve (base R)
actual <- test$diagnosis == "malignant"
thr <- sort(unique(c(0, prob, 1)), decreasing = TRUE)
tpr <- sapply(thr, function(t) mean(prob[actual] >= t))
fpr <- sapply(thr, function(t) mean(prob[!actual] >= t))
plot(fpr, tpr, type = "l", lwd = 2, col = "blue",
     main = "ROC Curve", xlab = "False Positive Rate",
     ylab = "True Positive Rate")
abline(0, 1, lty = 2)
auc <- sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1)) / 2)
cat("AUC :", round(auc, 4), "\n")
