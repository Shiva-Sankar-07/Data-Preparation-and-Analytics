## Install the necessary packages ##

install.packages('readr')     ## abalone.data is a large dataset. So, I've used readr package in dealing with that
install.packages('knitr')     ## To convert the r script into the markdown ans later for presentation, Knit is used for documentation.
install.packages('stringr')   ## It provides a cohesive set of functions designed to make working with strings as easy as possible
install.packages('caret')     ## To use machine learning models, I used caret to provide fit for our model
install.packages('corrplot')  ## With the corrplot I can provide the correlation matrix for our data.
install.packages('pROC')      ## For the ROC curves and analysis.

## These are the libraries that I'm using for the abalone data.
library(readr)
library(knitr)
library(stringr)
library(caret)
library(corrplot)
library(pROC)

# Reading the abalone data as the csv format
data_abal= read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',header = FALSE,sep = ",",stringsAsFactors = TRUE)

# Remove the observations in the Infant by keeping the Male/Female classes
infant_remove = subset(data_abal,V1!='I')
infant_remove$V1 = factor(infant_remove$V1)
set.seed(1)

# With the usage off the caret package we split the data into 80% and 20%.
partition_data = createDataPartition(infant_remove$V1,p=0.2,list=FALSE)

# Dividing the test and train data by separating the columns.
# test data has the infant data with the data part
test_data = infant_remove[partition_data,]
# Train data is without that data part
train_data = infant_remove[-partition_data,]

# Fit a logistic regression using all feature variables using the generalized linear models
# Used glm to apply that model
log_regression = glm(V1~V2+V3+V4+V5+V6+V7+V8+V9,data=train_data,family = binomial)

# Summary for the above logistic regression
summary(log_regression)

cat("\n The null hypothesis can be avoided for the variables for which the predictions have a lower p-value")
cat("\n We can tell from the output that V3 and V6 are the important predictors.")

# Now we have to present the confidence intervals for the logistic regression
confint(log_regression)
cat("\n Confidence interval does not contain 0 for V6 but it does for V3. V6 has 95% chance that
+  predictor V6 falls between range 2.05920944 & 6.09531362 and we can reject the null hypothesis.")

# The type as response provides the predicted probabilities
predic1= predict(log_regression,test_data,type="response")

# Create a new variable for the male and female and this can help us in making the confusion matrix
predic = ifelse(predic1>=0.5,'M','F')

# Confusion matrix for predictor for the test dataset.
confusionMatrix(as.factor(predic),as.factor(test_data$V1))

# plotting the ROC curve for the predictor
plot(roc(test_data$V1,predic1))
cat("\n As we can see ROC curve is better for our model")
cat("hence it will predict better than selecting random value")
cat("Accuracy of the model is 0.5599")

# plotting the mixed Correlation plot for the model
corrplot.mixed(cor(infant_remove[,2:8]))

# Conclusion
cat("\n Given that they don't explain much, the strong correlation between all the variables demonstrates the classifier's poor performance")
cat("\n A good model has uncorrelated variables.")


