library(tidyverse)
library(ggplot2)
library(randomForest)
library(caret)

url <- "https://raw.githubusercontent.com/samyvivo/Heart_disease/refs/heads/main/Heart_disease_raw_data.txt"

cols = c("age",
         "sex",# 0 = female, 1 = male
         "cp", # chest pain
         # 1 = typical angina,
         # 2 = atypical angina,
         # 3 = non-anginal pain,
         # 4 = asymptomatic
         "trestbps", # resting blood pressure (in mm Hg)
         "chol", # serum cholestoral in mg/dl
         "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
         "restecg", # resting electrocardiographic results
         # 1 = normal
         # 2 = having ST-T wave abnormality
         # 3 = showing probable or definite left ventricular hypertrophy
         "thalach", # maximum heart rate achieved
         "exang",   # exercise induced angina, 1 = yes, 0 = no
         "oldpeak", # ST depression induced by exercise relative to rest
         "slope", # the slope of the peak exercise ST segment
         # 1 = upsloping
         # 2 = flat
         # 3 = downsloping
         "ca", # number of major vessels (0-3) colored by fluoroscopy
         "thal", # this is short of thallium heart scan
         # 3 = normal (no cold spots)
         # 6 = fixed defect (cold spots during rest and exercise)
         # 7 = reversible defect (when cold spots only appear during exercise)
         "hd" # (the predicted attribute) - diagnosis of heart disease
         # 0 if less than or equal to 50% diameter narrowing
         # 1 if greater than 50% diameter narrowing
)


##Reading dataset from the source
data <- read.csv(url, header = F , col.names = cols)

##Viewing data and structures
glimpse(data)
View(data)

##Cleaning the question mark from dataset
data[data == "?"]  <- NA

##turn data to factor and recode some values
data <- data %>% 
  mutate(sex=recode_factor(sex, "1"="M", "0"="F"))

data <- data %>% 
  mutate(hd=recode_factor(hd, "0"="Healthy",
                          "1"="UnHealthy",
                          "2"="UnHealthy",
                          "3"="UnHealthy",
                          "4"="UnHealthy"))

data <- data %>% 
  mutate(across(c(cp,fbs,restecg,exang,slope,ca,thal), as.factor))


## Handle missing values using Random Forest imputation
set.seed(42)
data_imputed <- rfImpute(hd ~ ., data = data, iter = 6)


## Split data into training and testing sets (80/20)
index <- createDataPartition(data_imputed$hd, p = 0.8, list = FALSE)
train_data <- data_imputed[index, ]
test_data <- data_imputed[-index, ]

## Train Random Forest model
rf_model <- randomForest(hd ~ ., 
                         data = train_data,
                         ntree = 1000,
                         mtry = round(sqrt(ncol(train_data))),
                         importance = TRUE,
                         proximity = TRUE
)


## Model Evaluation
# Predict on test set
predictions <- predict(rf_model, test_data)

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, test_data$hd)
conf_matrix


# Variable Importance Plot
varImpPlot(rf_model, 
           main = "Variable Importance",
           type = 2)

##Using the OOB Error Plot for determine the best value for number of trees (ntree)
oob_error_data <- data.frame(
  Trees = rep(1:nrow(rf_model$err.rate), times=3), 
  Error = c(rf_model$err.rate[, "OOB"],
            rf_model$err.rate[, "Healthy"],
            rf_model$err.rate[, "UnHealthy"]),
  Type = rep(c("OOB", "Healthy", "UnHealthy"), 
  each = nrow(rf_model$err.rate)))
  

## OOB Error Rate Plot
ggplot(oob_error_data, aes(x = Trees, y = Error)) +
  geom_line(aes(color = Type)) +
  labs(title="Random Forest Error Rates",
       x="Number of Trees",
       y="Error Rate") +
  theme_minimal()
## NOTE: The plot shows us after 500 trees the err.rate stabilized
## So the best value for ntree = 500



## Feature Importance (as data frame)
importance_df <- as.data.frame(importance(rf_model, type = 2))
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini), ]
importance_df



## find the best mtry for model
oob_values <- vector(length=10)
for(i in 1:10) {
  temp.model <- randomForest(hd ~ ., data=data_imputed, mtry=i, ntree=1000)
  oob_values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob_values


## find the minimum error
min(oob_values)

## find the optimal value for mtry...
which(oob_values == min(oob_values))

## create a model for proximities using the best value for mtry and ntree
rf_model_optimized <- randomForest(hd ~ ., 
                      data=data_imputed,
                      ntree=500, 
                      proximity=TRUE,
                      importance = TRUE, 
                      mtry=which(oob_values == min(oob_values)))


## create an MDS-plot to show how the samples are related to each other.
## Start by converting the proximity matrix into a distance matrix.
distance_matrix <- as.dist(1-rf_model_optimized$proximity)

mds_stuff <- cmdscale(distance_matrix, eig=TRUE, x.ret=TRUE)

## calculate the percentage of variation that each MDS axis accounts for...
mds_var_per <- round(mds_stuff$eig/sum(mds_stuff$eig)*100, 1)

## draw plot that shows the MDS axes and the variation:
mds_values <- mds_stuff$points
mds_data <- data.frame(Sample=rownames(mds_values),
                       X=mds_values[,1],
                       Y=mds_values[,2],
                       Status=data_imputed$hd)

ggplot(data=mds_data, aes(x=X, y=Y, label=Sample, shape)) + 
  geom_point(aes(color=Status, shape=Status, size=Status)) +
  theme_bw() +
  scale_size_manual(values = c("Healthy" = 2.2, "UnHealthy" = 2.2)) +
  scale_color_manual(values = c("Healthy" = "green", "UnHealthy" = "red")) +
  labs(title="MDS plot using (1 - Random Forest Proximities)",
       x=paste("MDS1 - ", mds_var_per[1], "%", sep=""),
       y=paste("MDS2 - ", mds_var_per[2], "%", sep=""))



  