setwd("C:/Users/Samsung/Desktop")
library(foreign)
library(RMOA)
library(stream)
library(mlbench)
library(MASS)
library(plyr)
library(graphics)
library(corrplot)

reset <- function(x){
  x$state <- 1L
  x$processed <- 0L
  x$finished <- FALSE
}

count <- function(x){
  all <- nrow(x$data)
  processed <- x$processed
  left <- (all - processed)
}

stream <- read.csv("plane.csv", sep= ",")
stream$Class <- as.factor(stream$Class)
size <- nrow(stream)
datastream <- datastream_dataframe(data=stream)

chunk <<- 100

turns <- ceiling(size/chunk)

result_rhdt <- vector('numeric')
result_nb <- vector('numeric')
result_ac <- vector('numeric')
result_obo <- vector('numeric')
result_obg <- vector('numeric')

sample <- datastream$get_points(chunk)
sample <- datastream_dataframe(data=sample)

## Random Hoeffending Tree ##

rhdt <- MOAoptions(model = "RandomHoeffdingTree")
rhdt <- RandomHoeffdingTree(control=rhdt)
model_rhdt <<- trainMOA(model = rhdt,
                        Class ~ .,
                        data = sample)
reset(sample)

## Naive Bayes ##

nb <- MOAoptions(model = "NaiveBayes")
nb <- NaiveBayes(control=nb)
model_nb <<- trainMOA(model = nb,
                      Class ~ .,
                      data = sample)
reset(sample)

## Active Classifier ##

ac <- MOAoptions(model = "ActiveClassifier")
ac <- ActiveClassifier(control=ac)
model_ac <<- trainMOA(model = ac,
                      Class ~ .,
                      data = sample)
reset(sample)

## Oza Boost ##

obo <- MOAoptions(model = "OzaBoost")
obo <- OzaBoost(control=obo)
model_obo <<- trainMOA(model = obo,
                       Class ~ .,
                       data = sample)
reset(sample)

## Oza Bag ##

obg <- MOAoptions(model = "OzaBag")
obg <- OzaBag(control=obg)
model_obg <<- trainMOA(model = obg,
                       Class ~ .,
                       data = sample)

list <- 1:turns
progress.bar <- create_progress_bar("text")
progress.bar$init(turns)

while ((count(datastream)) >= chunk ){
  sample <- datastream$get_points(chunk)
  
  ## Hoeffending Tree ##  
  
  scores <- predict(model_rhdt,
                    newdata=sample[, colnames(sample[1:12])],
                    type="response")
  table(scores, sample$Class)
  chunk_acc_rhdt <- mean((scores == sample$Class)*100)
  result_rhdt <- append(result_rhdt, chunk_acc_rhdt)
  
  ## Naive Bayes ##
  
  scores <- predict(model_nb,
                    newdata=sample[, colnames(sample[1:12])],
                    type="response")
  table(scores, sample$Class)
  chunk_acc_nb <- mean((scores == sample$Class)*100)
  result_nb <- append(result_nb, chunk_acc_nb)
  
  ## AC ##
  
  scores <- predict(model_ac,
                    newdata=sample[, colnames(sample[1:12])],
                    type="response")
  table(scores, sample$Class)
  chunk_acc_ac <- mean((scores == sample$Class)*100)
  result_ac <- append(result_ac, chunk_acc_ac)
  
  ## Oza Boost ##
  
  scores <- predict(model_obo,
                    newdata=sample[, colnames(sample[1:12])],
                    type="response")
  table(scores, sample$Class)
  chunk_acc_obo <- mean((scores == sample$Class)*100)
  result_obo <- append(result_obo, chunk_acc_obo)
  
  ## Oza Bag ##
  
  scores <- predict(model_obg,
                    newdata=sample[, colnames(sample[1:12])],
                    type="response")
  table(scores, sample$Class)
  chunk_acc_obg <- mean((scores == sample$Class)*100)
  result_obg <- append(result_obg, chunk_acc_obg)
  
  ###########################################################
  ###########################################################
  
  sample <- datastream_dataframe(sample)
  
  ## Hoeffending Tree ## 
  
  model_rhdt <- trainMOA(model = model_rhdt$model, 
                         formula = Class ~., 
                         data = sample,
                         reset=FALSE,
                         trace=FALSE)
  reset(sample)
  
  ## Naive Bayes ##
  
  model_nb <- trainMOA(model = model_nb$model, 
                       formula = Class ~., 
                       data = sample,
                       reset=FALSE,
                       trace=FALSE)   
  reset(sample)
  
  ## Active Classifier ##
  
  model_ac <- trainMOA(model = model_ac$model, 
                       formula = Class ~., 
                       data = sample,
                       reset=FALSE,
                       trace=FALSE)   
  reset(sample)
  
  ## Oza Boost ##
  
  model_obo <- trainMOA(model = model_obo$model, 
                        formula = Class ~., 
                        data = sample,
                        reset=FALSE,
                        trace=FALSE)
  reset(sample)
  
  ## Oza Bag ##
  
  model_obg <- trainMOA(model = model_obg$model, 
                        formula = Class ~., 
                        data = sample,
                        reset=FALSE,
                        trace=FALSE)
  
  progress.bar$step()
}

result_rhdt
result_nb
result_ac
result_obo
result_obg
