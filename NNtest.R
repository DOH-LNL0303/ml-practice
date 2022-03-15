# Example script for neural networks
# https://www.geeksforgeeks.org/building-a-simple-neural-network-in-r-programming/


# Package names
packages <- c("dplyr", "neuralnet", "tidytext", "wesanderson", "stringr", "caret")

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE))

setwd("C:/R Work Folder/Machine Learning")
getwd()
data <- read.csv(file = 'data/binary.csv')
str(data)

## Step 1: Scaling the Data

# Draw a histogram for gre data
hist(data$gre)

# # The min-max normalization transforms the data into a common range, 
# thus removing the scaling effect from all the variables.

# Normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


# Min-Max Normalization so that scale is now 0 to 1
data$gre <- (data$gre - min(data$gre)) / (max(data$gre) - min(data$gre))
hist(data$gre) #ta - da! Normal distribution

# Min-Max Normalization
data$gpa <- (data$gpa - min(data$gpa)) / (max(data$gpa) - min(data$gpa))
hist(data$gpa)

data$rank <- (data$rank - min(data$rank)) / (max(data$rank) - min(data$rank))
hist(data$rank)

## Step 2: Sampling of the data

# Divide the data into a training set and test set. 
# The training set is used to find the relationship between dependent and independent 
# variables while the test set analyses the performance of the model. 
# We use 70% of the dataset as a training set.

set.seed(222) # gen random sample
inp <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
training_data <- data[inp==1, ]
test_data <- data[inp==2, ]

## Step 3: Fitting a Neural Network

# Formula
#neuralnet(formula, data, hidden = 1, stepmax = 1e+05, rep = 1, lifesign = “none”, 
          #algorithm = “rprop+”, err.fct = “sse”, linear.output = TRUE)

set.seed(333)
n <- neuralnet(admit~gre + gpa + rank, #formula
               data = training_data,
               hidden = 5, # number of hidden layers
               err.fct = "ce", # error calculation, here cross-entropy
               linear.output = FALSE,
               lifesign = 'full',
               rep = 2, # number of reps for neural net training
               algorithm = "rprop+", # resilient backpropagation with weighting
               stepmax = 100000) # max steps until model full stop


# Did not converge (rep 2), but rep 1 has less error. So we're going with that!
# Visualize


# plot our neural network 
plot(n, rep = 1)

# The model has 5 neurons in its hidden layer. The black lines show the connections with weights.
# The weights are calculated using the backpropagation algorithm. 
# The blue line is displays the bias term (constant in a regression equation).


# error
n$result.matrix

# Step 4: Prediction
# Don't forget that the prediction rating will be scaled

# Prediction
output <- compute(n, rep = 1, training_data[, -1])
# compare predicted rating with real rating

# predicted
head(output$net.result)
# real
head(training_data[1, ])

## Step 5: Confusion Matrix and Misclassification error


# confusion Matrix $Misclassification error -Training data
output <- compute(n, rep = 1, training_data[, -1])
p1 <- output$net.result
pred1 <- ifelse(p1 > 0.5, 1, 0)

tab1 <- table(pred1, training_data$admit) #xtab of predicted versus observed in training data
tab1

# Calculate misclassification error
1 - sum(diag(tab1)) / sum(tab1)

# 25.6%

pred2 <- as.data.frame(pred1)
pred2$yes <- ifelse(p1 > 0.5, 1, 0)

pred2 <- pred2 %>% 
  filter(yes==1)

pred2 <- as.vector(pred2)
class(pred2)
  

example <- confusionMatrix(data = pred1$yes, reference = training_data$admit)

# # Next steps
# We can further increase the accuracy and efficiency of our model by
# increasing of decreasing nodes and bias in hidden layers.

# Questions
#1. guidance on selecting algorithm and error prediction?
#2. what is the relationship between model performance and number of hidden layers?
# 3. 
