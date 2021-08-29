# Install packages if required and load them
if(!require(caret)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

library(caret)
library(tidyverse)

# Download the data file and rename columns for better interpretability
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data", dl)
data <- read.csv(dl) %>% 
  rename("Time since last donation (months)" = "Recency..months.") %>%
  rename("Total number of blood donations" = "Frequency..times.") %>%
  rename("Amount of blood donated (mL)" = "Monetary..c.c..blood.") %>%
  rename("Time since first donation (months)" = "Time..months.") %>%
  rename("Donated Blood in March 2007?" = "whether.he.she.donated.blood.in.March.2007")

# Partitioning of data into training and test sets
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(data$`Donated Blood`, times = 1, p = 0.1, list = FALSE)
training <- data[-test_index,] %>% select(-`Amount of blood donated (mL)`)
validation <- data[test_index,] %>% select(-`Amount of blood donated (mL)`)


test_index <- createDataPartition(training$`Donated Blood`, times = 1, p = 0.1, list = FALSE)
train <- training[-test_index,]
test_set <- training[test_index,]
rm(test_index)

## Training dataset without "Amount of blood donated (mL)" filtered out. Used to demonstrate
## collinearity of Total number of blood donations and Amount of blood donated (mL) variables
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(data$`Donated Blood`, times = 1, p = 0.1, list = FALSE)
training1 <- data[-test_index,]

test_index <- createDataPartition(training1$`Donated Blood`, times = 1, p = 0.1, list = FALSE)
train1 <- training1[-test_index,]
rm(test_index)

#Defining a function to calculate the root mean squared error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))
}

#Exploratory Data Analysis
## Histograms of variables in the training data set
training %>% ggplot(aes(x = `Time since last donation (months)`)) +
  geom_histogram() +
  ylab("Count")

training %>% ggplot(aes(x = `Total number of blood donations`)) +
  geom_histogram() +
  ylab("Count")

training %>% ggplot(aes(x = `Time since first donation (months)`)) +
  geom_histogram() +
  ylab("Count")

training %>% ggplot(aes(x = `Donated Blood in March 2007?`)) +
  geom_bar() +
  ylab("Count")

## Summary statistics of variables in the training data set
training %>%
  summarize(Mean = mean(`Time since last donation (months)`),
            Median = median(`Time since last donation (months)`),
            Min = min(`Time since last donation (months)`),
            Max = max(`Time since last donation (months)`),
            Range = max(`Time since last donation (months)`) - min(`Time since last donation (months)`)) %>%
  knitr::kable(format.args = list(big.mark = ","), caption = "Summary Statistics of \"Time Since Last Donation (months)\" in the Training Data Set")

training %>%
  summarize(Mean = mean(`Total number of blood donations`),
            Median = median(`Total number of blood donations`),
            Min = min(`Total number of blood donations`),
            Max = max(`Total number of blood donations`),
            Range = max(`Total number of blood donations`) - min(`Total number of blood donations`)) %>%
  knitr::kable(format.args = list(big.mark = ","), caption = "Summary Statistics of \"Total Number of Blood Donations\" in the Training Data Set")

training %>%
  summarize(Mean = mean(`Amount of blood donated (mL)`),
            Median = median(`Amount of blood donated (mL)`),
            Min = min(`Amount of blood donated (mL)`),
            Max = max(`Amount of blood donated (mL)`),
            Range = max(`Amount of blood donated (mL)`) - min(`Amount of blood donated (mL)`)) %>%
  knitr::kable(format.args = list(big.mark = ","), caption = "Summary Statistics of \"Amount of Blood Donated (mL)\" in the Training Data Set")

training %>%
  summarize(Mean = mean(`Time since first donation (months)`),
            Median = median(`Time since first donation (months)`),
            Min = min(`Time since first donation (months)`),
            Max = max(`Time since first donation (months)`),
            Range = max(`Time since first donation (months)`) - min(`Time since first donation (months)`)) %>%
  knitr::kable(format.args = list(big.mark = ","), caption = "Summary Statistics of \"Time Since First Donation (months)\" in the Training Data Set")

train %>%
  group_by(`Donated Blood in March 2007?`) %>%
  summarize(n()) %>%
  rename("n" = `n()`) %>%
  knitr::kable(format.args = list(big.mark = ","), caption = "Summary Statistics of Blood Donation Status in March 2007 in the Training Data Set")


## Graphs
train %>% 
  group_by(`Time since last donation (months)`) %>%
  summarize(prop = mean(`Donated Blood in March 2007?` == 1)) %>%
  ggplot(aes(x = `Time since last donation (months)`, y = `prop`)) +
  geom_point()

train %>% 
  group_by(`Total number of blood donations`) %>%
  summarize(prop = mean(`Donated Blood in March 2007?` == 1)) %>%
  ggplot(aes(x = `Total number of blood donations`, y = `prop`)) +
  geom_point()

train %>% 
  group_by(`Time since first donation (months)`) %>%
  summarize(prop = mean(`Donated Blood in March 2007?` == 1)) %>%
  ggplot(aes(x = `Time since first donation (months)`, y = `prop`)) +
  geom_point()

train1 %>%
  ggplot(aes(x = `Amount of blood donated (mL)`, y = `Total number of blood donations`)) + geom_point()

# Models
# Model 1: Flip a coin by sampling from 0 and 1 with 50% probability of each
yhat <- sample(c(0,1), length(test_set$`Donated Blood in March 2007?`), replace = TRUE)

model1_accuracy <- mean(yhat == test_set$`Donated Blood in March 2007?`)
model1_rmse <- RMSE(yhat, test_set$`Donated Blood in March 2007?`)
model1_F1 <- F_meas(as.factor(yhat), reference = as.factor(test_set$`Donated Blood in March 2007?`))

# Model 2: Logistic Regression
fit_glm <- glm(`Donated Blood in March 2007?` ~ ., data = train, family = "binomial")
p_hat_glm <- predict(fit_glm, test_set, type = "response")
y_hat_glm <- ifelse(p_hat_glm > 0.5, "1", "0") %>% factor(levels = levels(as.factor(test_set$`Donated Blood in March 2007?`)))

model2_accuracy <- confusionMatrix(y_hat_glm, as.factor(test_set$`Donated Blood in March 2007?`))$overall[["Accuracy"]]
model2_rmse <- RMSE(as.numeric(y_hat_glm), as.numeric(test_set$`Donated Blood in March 2007?`))
model2_F1 <- F_meas(y_hat_glm, reference = as.factor(test_set$`Donated Blood in March 2007?`))

# Model 3: k-Nearest Neighbours
train_knn <- train(as.factor(`Donated Blood in March 2007?`) ~ ., method = "knn", data = train, tuneGrid = data.frame(k = seq(5,100,2)))
y_hat_knn <- predict(train_knn, test_set, type = "raw")

model3_accuracy <- confusionMatrix(y_hat_knn, as.factor(test_set$`Donated Blood in March 2007?`))$overall["Accuracy"]
model3_rmse <- RMSE(as.numeric(y_hat_knn), test_set$`Donated Blood in March 2007?`)
model3_F1 <- F_meas(predict(train_knn, test_set, type = "raw"), reference = as.factor(test_set$`Donated Blood in March 2007?`))       

## kNN accuracy versus k graph
qplot(x = train_knn$results$k, y = train_knn$results$Accuracy) +
  xlab("Number of Nearest Neighbours k") +
  ylab("Accuracy on training set \"train\"") +
  scale_x_continuous(limits = c(0,NA))

# Model 4: Random Forest
train_rf <- train(as.factor(`Donated Blood in March 2007?`) ~ ., method = "rf", data = train, tuneGrid = data.frame(mtry = c(2,3,4)))
y_hat_rf <- predict(train_rf, test_set, type = "raw")
  
model4_accuracy <- confusionMatrix(y_hat_rf, as.factor(test_set$`Donated Blood in March 2007?`))$overall["Accuracy"]
model4_rmse <- RMSE(as.numeric(y_hat_rf), as.numeric(test_set$`Donated Blood in March 2007?`))
model4_F1 <- F_meas(predict(train_rf, test_set, type = "raw"), reference = as.factor(test_set$`Donated Blood in March 2007?`)) 

## Random forest accuracy versus number of randomly-selected predictors graph
qplot(x = train_rf$results$mtry, y = train_rf$results$Accuracy) +
  xlab("Number of Randomly Selected Predictors") +
  ylab("Accuracy (Bootstrap)") +
  scale_x_continuous(limits = c(0, NA))

# Model 5: Ensemble of k Nearest Neighbours and Random Forest
p_hat_knn <- predict(train_knn, test_set, type = "prob")
p_hat_rf <- predict(train_rf, test_set, type = "prob")
p_hat_knn_rf <- (p_hat_knn + p_hat_rf)/2
y_hat_knn_rf <- factor(apply(p_hat_knn_rf, 1, which.max)-1)

model5_accuracy <- confusionMatrix(y_hat_knn_rf, as.factor(test_set$`Donated Blood in March 2007?`))$overall["Accuracy"]
model5_rmse <- RMSE(as.numeric(y_hat_knn_rf), as.numeric(test_set$`Donated Blood in March 2007?`))
model5_F1 <- F_meas(y_hat_knn_rf, reference = as.factor(test_set$`Donated Blood in March 2007?`))

# Validation of the final model (Model 3) against the hold-out test set
y_hat_knn_validation <- predict(train_knn, validation, type = "raw")

validation_accuracy <- confusionMatrix(y_hat_knn_validation, as.factor(validation$`Donated Blood in March 2007?`))$overall["Accuracy"]
validation_rmse <- RMSE(as.numeric(y_hat_knn_validation), validation$`Donated Blood in March 2007?`)
validation_F1 <- F_meas(y_hat_knn_validation, reference = as.factor(validation$`Donated Blood in March 2007?`))

# Results tables
## Model 1
rmse_results1 <- tibble("Model" = "1: Flip a coin", "RMSE" = model1_rmse, "Accuracy" = model1_accuracy, "F1 score" = model1_F1)
rmse_results1 %>% knitr::kable()

## Model 2
rmse_results2 <- tibble("Model" = "2: Logistic Regression", "RMSE" = model2_rmse, "Accuracy" = model2_accuracy, "F1 score" = model2_F1)
rbind(rmse_results1, rmse_results2) %>% knitr::kable()

## Model 3
rmse_results3 <- tibble("Model" = "3: k-Nearest Neighbours", "RMSE" = model3_rmse, "Accuracy" = model3_accuracy, "F1 score" = model3_F1)
rbind(rmse_results1, rmse_results2, rmse_results3) %>% knitr::kable()

## Model 4
rmse_results4 <- tibble("Model" = "4: Random Forest", "RMSE" = model4_rmse, "Accuracy" = model4_accuracy, "F1 score" = model4_F1)
rbind(rmse_results1, rmse_results2, rmse_results3, rmse_results4) %>% knitr::kable()

## Model 5
rmse_results5 <- tibble("Model" = "5: Ensemble of kNN and Random Forest", "RMSE" = model5_rmse, "Accuracy" = model5_accuracy, "F1 score" = model5_F1)
rbind(rmse_results1, rmse_results2, rmse_results3, rmse_results4, rmse_results5) %>% knitr::kable()

## Validation
rmse_results_validation <- tibble("Model" = "Validation using hold-out test set `validation`", "RMSE" = validation_rmse, "Accuracy" = validation_accuracy, "F1 score" = validation_F1)
rbind(rmse_results3, rmse_results_validation) %>% knitr::kable()