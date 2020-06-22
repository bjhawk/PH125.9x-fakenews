library(caret)
library(tidytext)
library(tidyr)
library(data.table)
library(dplyr)
library(dtplyr)
library(stringr)
library(kernlab)
library(doSNOW)

## e1071, randomForest, foreach, import

remove(list = ls())
gc()
options(digits = 3)
set.seed(1989)

## SET THIS CAREFULLY
nThreads <- 16

# Read data
real.news <- fread(file.path('./data', 'True.csv'))
real.news[, is_fake := FALSE]

fake.news <- fread(file.path('./data', 'Fake.csv'))
fake.news[, is_fake := TRUE]

all.news <- rbind(real.news, fake.news)

remove(real.news, fake.news)
message("Cleaning Dataset, this takes a few moments (iconv + grep)")
# Clean and tidy dataset
all.news <- lazy_dt(all.news) %>%
  mutate(
    title = str_trim(iconv(title, from = "utf8", to = "latin1")),
    text = str_trim(iconv(text, from = "utf8", to = "latin1"))
  ) %>%
  filter(!is.na(title) & title != '') %>%
  mutate(
    full_text = paste(text, title),
    is_fake = as.factor(is_fake),
    title_caps = str_count(title, "[^a-z][A-Z]+[^a-z]")
  ) %>%
  select(title, full_text, is_fake, title_caps) %>%
  as.data.table()

# Mark training and test datasets, they will be split later.
train.index <- createDataPartition(
  all.news$is_fake,
  p = 0.8,
  times = 1,
  list = FALSE
)

all.news$set <- "testing"
all.news[train.index, set := "training"]

remove(train.index)

message("Splitting tokens")
# This takes a few moments
timer <- proc.time()
# split tokens for joining sentiment, remove stop words.
tokenized <- all.news %>%
  unnest_tokens(token, full_text) %>%
  lazy_dt() %>%
  anti_join(data.table(token = stop_words$word), by = "token") %>%
  as.data.table()
message("Tokens split in ", timer - proc.time(), " seconds.")

remove(all.news)

message("joining dataset to sentiments")

## Note that inner joins will inherently filter out tokens
## not present in the given lexicon

message(". . . Afinn")
timer <- proc.time()
afinn <- fread("./data/afinn.csv")

# Change names of columns for joining
# using as_tibble > inner join > as.data.table due to discrepencies
# in how joins are implemented in the data.table dtplyr backend
setnames(afinn, c("token", "sentiment"))
afinn <- as_tibble(tokenized) %>%
  inner_join(afinn, by = "token") %>%
  as.data.table()

# aggregate total sentiment for each article
afinn <- afinn[
  ,
  list(sentiment = sum(sentiment)),
  by  = list(title, is_fake, title_caps, set)
]

message("Afinn sentiment aggregated in ", timer - proc.time(), " seconds.")

message(". . . NRC")
timer <- proc.time()
nrc <- fread("./data/nrc.csv")

setnames(nrc, "word", "token")
setkey(nrc, token)
setkey(tokenized, token)

nrc <- as_tibble(tokenized) %>%
  inner_join(nrc, by = "token") %>%
  as.data.table()

# Tibbles with pivot_wider is a much easier-to-read approach here.
# but there are other more performant ways of doing this if our
# dataset was very large.
nrc <- as_tibble(nrc) %>%
  pivot_wider(
    names_from = sentiment,
    values_from = sentiment,
    values_fn = list(sentiment = length),
    values_fill = list(sentiment = 0)
  ) %>%
  group_by(title, is_fake, title_caps, set) %>%
  summarize_at(vars(-token), list(sum)) %>%
  as.data.table()

message("NRC sentiment aggregated in ", timer - proc.time(), " seconds.")

message(". . . NRC VAD")
timer <- proc.time()
vad <- fread("./data/nrc_vad.csv")
setnames(vad, tolower(names(vad)))
setnames(vad, "word", "token")

setkey(vad, token)
setkey(tokenized, token)
vad <- as_tibble(tokenized) %>%
  inner_join(vad, by = "token") %>%
  as.data.table()

# roll up - sum V/A/D
vad <- vad[
  ,
  list(
    valence = sum(valence),
    arousal = sum(arousal),
    dominance = sum(dominance)
  ),
  by = list(title, is_fake, title_caps, set)
]
message("NRC sentiment aggregated in ", timer - proc.time(), " seconds.")

## Splitting test and training sets
afinn <- split(afinn, by = "set", keep.by = FALSE)
afinn.training <- afinn$training
afinn.testing <- afinn$testing

nrc <- split(nrc, by = "set", keep.by = FALSE)
nrc.training <- nrc$training
nrc.testing <- nrc$testing

vad <- split(vad, by = "set", keep.by = FALSE)
vad.training <- vad$training
vad.testing <- vad$testing

remove(tokenized, afinn, nrc, vad)

## Begin building models

# We can leveraging matrix-based function signatures for all the models we build
# This helper function will create a matrix of all predictors from a given data.table
makePredictors <- function(dt) {
  # drop our title, used only as an identifier column
  dt$title = NULL

  # drop the response column
  dt$is_fake = NULL

  as.matrix(dt)
}

## Random Forests
cl <- makeSOCKcluster(nThreads)
registerDoSNOW(cl)

rf.trainControl <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE
)

# RF AFINN
message("RF Afinn")
timer <- proc.time()
afinn.rf.model <- train(
  makePredictors(afinn.training),
  afinn.training$is_fake,
  method = "parRF",
  trControl = rf.trainControl
)
message("Afinn Random Forest built in ", timer - proc.time(), " seconds.")

# RF NRC
message("RF NRC")
timer <- proc.time()
nrc.rf.model <- train(
  makePredictors(nrc.training),
  nrc.training$is_fake,
  method = "parRF",
  trControl = rf.trainControl
)
message("NRC Random Forest built in ", timer - proc.time(), " seconds.")

# RF VAD
message("RF NRC VAD")
timer <- proc.time()
vad.rf.model <- train(
  makePredictors(vad.training),
  vad.training$is_fake,
  method = "parRF",
  trControl = rf.trainControl
)
message("NRC VAD Random Forest built in ", timer - proc.time(), " seconds.")

## Radial KSVMs

# We defualt to C = 1
# as the max penalty for large residuals, since our dataset is relatively small

ksvm.trainControl <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE
)

# KSVM AFINN
message("KSVM Afinn")
timer <- proc.time()
afinn.training.predictors <- makePredictors(afinn.training)

afinn.training.sigmas <- sigest(afinn.training.predictors, frac = 1)
afinn.training.sigmas <- seq(
  afinn.training.sigmas["90%"] * 0.75,
  afinn.training.sigmas["10%"] * 1.25,
  length.out = 10
)

afinn.ksvm.model <- train(
  afinn.training.predictors,
  afinn.training$is_fake,
  method = 'svmRadial',
  trControl = ksvm.trainControl,
  tuneGrid = data.table(
    sigma = afinn.training.sigmas,
    C = 1
  )
)
message("Afinn SVM model built in ", timer - proc.time(), " seconds.")

# KSVM NRC
message("KSVM NRC")
timer <- proc.time()
nrc.training.predictors <- makePredictors(nrc.training)

nrc.training.sigmas <- sigest(nrc.training.predictors, frac = 1)
nrc.training.sigmas <- seq(
  nrc.training.sigmas["90%"] * 0.75,
  nrc.training.sigmas["10%"] * 1.25,
  length.out = 10
)

nrc.ksvm.model <- train(
  nrc.training.predictors,
  nrc.training$is_fake,
  method = 'svmRadial',
  trControl = ksvm.trainControl,
  tuneGrid = data.table(
    sigma = nrc.training.sigmas,
    C = 1
  )
)
message("NRC SVM model built in ", timer - proc.time(), " seconds.")

# KSVM VAD
message("KSVM VAD")
timer <- proc.time()
vad.training.predictors <- makePredictors(vad.training)

vad.training.sigmas <- sigest(vad.training.predictors, frac = 1)
vad.training.sigmas <- seq(
  vad.training.sigmas["90%"] * 0.75,
  vad.training.sigmas["10%"] * 1.25,
  length.out = 10
)

vad.ksvm.model <- train(
  vad.training.predictors,
  vad.training$is_fake,
  method = 'svmRadial',
  trControl = ksvm.trainControl,
  tuneGrid = data.table(
    sigma = vad.training.sigmas,
    C = 1
  )
)
message("NRC VAD SVM model built in ", timer - proc.time(), " seconds.")

## Stop and deregister parallel computing
stopCluster(cl)
registerDoSEQ()
remove(cl)

## Accuracy measures against training datasets
# Afinn RF
confusionMatrix(fitted(afinn.rf.model), afinn.training$is_fake)
# 74.9
# 82.3

# NRC RF
confusionMatrix(fitted(nrc.rf.model), nrc.training$is_fake)
# 99.2
# 99.8

# NRC VAD RF
confusionMatrix(fitted(vad.rf.model), vad.training$is_fake)
# 99.8
# 99.8


# Afinn SVM
confusionMatrix(fitted(afinn.ksvm.model$finalModel), afinn.training$is_fake)
# 74.8
# 82

# NRC SVM
confusionMatrix(fitted(nrc.ksvm.model$finalModel), nrc.training$is_fake)
# 83.6
# 88.4

# NRC VAD SVM
confusionMatrix(fitted(vad.ksvm.model$finalModel), vad.training$is_fake)
# 80.8
# 86.2

## Make final predictions/measure Acc
# Afinn RF
confusionMatrix(predict(afinn.rf.model, afinn.testing), afinn.testing$is_fake)
# 76.6
# 81.9

# NRC RF
confusionMatrix(predict(nrc.rf.model, nrc.testing), nrc.testing$is_fake)
# 84.3
# 89

# NRC VAD RF
confusionMatrix(predict(vad.rf.model, vad.testing), vad.testing$is_fake)
# 83.3
# 87.2


# Afinn SVM
confusionMatrix(predict(afinn.ksvm.model, makePredictors(afinn.testing)), afinn.testing$is_fake)
# 77
# 82.3

# NRC SVM
confusionMatrix(predict(nrc.ksvm.model, makePredictors(nrc.testing)), nrc.testing$is_fake)
# 80.7
# 87.3

# NRC VAD SVM
confusionMatrix(predict(vad.ksvm.model, makePredictors(vad.testing)), vad.testing$is_fake)
# 80.8
# 86.1

## Ensembles
# training

# testing
