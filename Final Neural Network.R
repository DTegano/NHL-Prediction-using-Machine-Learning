### Final Neural Network Model ###

library(readxl) # importing xls & xlsx files
library(dplyr) # used for nesting/chaining
library(tidyr) # data cleaning
library(stringr) # Used to detect OT and Shootouts
library(writexl) # exporting to xlsx file
library(chron) # Date conversion to Time
library(stats) # Group Functional
library(plyr) # ddpylr for aggregation
library(caret) # Confusion Matrix
library(gmodels) #cross table
library(ggplot2)
library(cowplot)
library(tidyverse)
library(broom)
library(kernlab) #used for SVM
library(neuralnet) #used for ANN
library(keras)
library(tensorflow)
library(deepviz)
library(magrittr)

# Set up Directories
getwd()

setwd("C:/Users/David/Documents/MSDS 696/data")

# Import Data
dtrain = read_excel("Training Final CM.xlsx", col_names = TRUE)
dtest = read_excel("Test Final CM.xlsx", col_names = TRUE)

ttrain = read_excel("Training Final.xlsx", col_names = TRUE)
ttest = read_excel("Test Final.xlsx", col_names = TRUE)

#NEW DATA if better results - no
dtrain = read_excel("Training.xlsx", col_names = TRUE)
dtest = read_excel("Test.xlsx", col_names = TRUE)

# Remove results from test set
test = dtest[,-4]
atest = ttest[,-4]

# additional Training Data
add_data =  read_excel("2015-2017 Add Training.xlsx", col_names = TRUE)
dtrain = rbind(add_data, dtrain)

# Remove Team Names and Data from training and test sets
dtrain = dtrain[, -c(1:3, 37,38)] #dont need games played with cm
ttrain = ttrain[, -c(1:3, 37,38)]
dtest = dtest[, -c(1:3, 37,38)]
ttest = ttest[, -c(1:3, 37,38)]

# Convert data to factors
dtrain$Result = as.factor(dtrain$Result)
ttrain$Result = as.factor(ttrain$Result)
dtest$Result = as.factor(dtest$Result)
ttest$Result = as.factor(ttest$Result)

dt = rbind(dtrain, dtest)

# Normalize/Scale Function
normaliz = function(x) {
  (x-min(x))/(max(x) - min(x))
}

scal = function(x) {
  scale(x, center = TRUE, scale = TRUE)
}

# Set up Model
dtrain$Result = as.numeric(dtrain$Result)
dtest$Result = as.numeric(dtest$Result)

train = as.matrix(dtrain)
test = as.matrix(dtest)

dimnames(train) = NULL
dimnames(test) = NULL

train[,2:45] = scal(train[,2:45]) # 53 with new data
test[,2:45] = scal(test[,2:45])

train[,1] = train[,1] - 1
test[,1] = test[,1] - 1

traintarget = train[,1]
testtarget = test[,1]

train = train[, 2:45]
test = test[, 2:45]

trainLabels = to_categorical(traintarget) #check tf_config if issue
testLabels = to_categorical(testtarget)

head(testLabels)

## Neural Network Model

set.seed(18)

# API model

model = NULL

model <- local({
  input = layer_input(shape = c(44), name = 'main_input')
  
  layer1 = input %>%
    layer_dense(units = 30, activation = "relu")
  
  layer2 = input %>%
    layer_dense(units = 30, activation = "relu")
  
  output = layer_concatenate(c(layer1, layer2)) %>%
    layer_dropout(0.5) %>%
    layer_batch_normalization() %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = 2, activation = "sigmoid", name = 'main_outout')
  
  keras_model(inputs = input, outputs = output)
})

summary(model)

# plot architecture
model %>% plot_model()

# compile
model %>%
  compile(loss = "binary_crossentropy",
          optimizer = optimizer_adam(lr=.0001),
          metrics = "accuracy")

history = model %>%
  fit(train,
      trainLabels,
      epoch = 200,
      batch_size = 100,
      validation_split = 0.2)

plot(history)

# evaluate model
model %>%
  evaluate(test, testLabels)

max_val_acc  = order(history$metrics$val_accuracy, decreasing = TRUE)
epoch = max_val_acc[1] 
epoch
history$metrics$val_accuracy[epoch]

prob = model %>%
  predict_on_batch(test)

pred = ifelse(model$predict(test)[,1]>model$predict(test)[,2], 0,1)

table(Predicted = pred, Actual = testtarget)

Results = ifelse(pred==1, "W", "L")
Results = as.factor(Results)

confusionMatrix(ttest$Result, Results, positive = "W")


#save and load weights
model$load_weights("C:/Users/David/Documents/MSDS 696/model")

wts = model$weights

# pull best val accuracy
max_val_acc  = order(history$metrics$val_accuracy, decreasing = TRUE)
epoch = max_val_acc[1] 
epoch
history$metrics$val_accuracy[epoch]

# plot val acc
plot(history$metrics$val_accuracy, col = 'blue')
max_val_acc  = order(history$metrics$val_accuracy, decreasing = TRUE)
epoch = max_val_acc[1] 
epoch
points(epoch, max(history$metrics$val_accuracy), col = 'green')

# reg line
mod = lm(history$metrics$val_accuracy ~ index)
abline(mod, col = 'red')

mod = glm(history$metrics$val_accuracy ~ poly(index, degree = 2))
lines(index, predict(mod), col = 'red')

#or 
lines(lowess(history$metrics$val_accuracy ~ index))
lines(lowess(history$metrics$val_accuracy))
# save weights??

checkpoint_path = "checkpoints/cp.ckpt"

cp_callback = callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  verbose = 0
  )

history = model %>%
  fit(train,
      trainLabels,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.20,
      callbacks = list(cp_callback),
      verbose = 2)

model %>%
  load_model_weights_tf(filepath = checkpoint_path)

callback_model_checkpoint(
  filepath = "C:/Users/David/Documents/MSDS 696/model/call",
  monitor = "val_loss",
  verbose = 0,
  save_best_only = TRUE,
  save_weights_only = FALSE,
  mode = 'min',
  period = NULL,
  save_freq = "epoch"
)

# manual K_fold - get train and hotcode first
train_2015 = train[1:1211,]
train_2016 = train[1212:2423,]
train_2017 = train[2424:3746,]
train_2018 = train[3747:5017,]

label_2015 = trainLabels[1:1211,]
label_2016 = trainLabels[1212:2423,]
label_2017 = trainLabels[2424:3746,]
label_2018 = trainLabels[3747:5017,]



#1st fold
train_1 = rbind(train_2018,train_2017,train_2016,train_2015)
label_1 = rbind(label_2018,label_2017,label_2016,label_2015)

history = model %>%
  fit(train_1,
      label_1,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.24)

#2nd fold
train_2 = rbind(train_2015,train_2017,train_2016,train_2018)
label_2 = rbind(label_2015,label_2017,label_2016,label_2018)

history = model %>%
  fit(train_2,
      label_2,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.25)

#3rd fold
train_3 = rbind(train_2016,train_2018,train_2015,train_2017)
label_3 = rbind(label_2016,label_2018,label_2015,label_2017)

history = model %>%
  fit(train_3,
      label_3,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.26)

#4th fold
train_4 = rbind(train_2017,train_2015,train_2018,train_2016)
label_4 = rbind(label_2017,label_2015,label_2018,label_2016)

history = model %>%
  fit(train_4,
      label_4,
      epoch = 50,
      batch_size = 100,
      validation_split = 0.24)