library(caret)
library(tidytext)
library(tidyr)
library(data.table)
library(dplyr)
library(dtplyr)
library(stringr)
library(kernlab)
library(doSNOW)

## TODO what libs need to be required/installed?
## e1071, randomForest, foreach, import

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

# Clean and tidy dataset
# + Fix character encoding, trim whitespace
# + Remove observations without a title
# + Create a "full_text" column
# + Make outcome column a factor
# + Add a column counting the number of capitalized words in the title
message("Cleaning Dataset, this takes a few moments (iconv + grep)")
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
timer <- proc.time()

# split tokens for joining sentiment, remove stop words.
# This takes a few moments
tokenized <- all.news %>%
  unnest_tokens(token, full_text) %>%
  lazy_dt() %>%
  anti_join(data.table(token = stop_words$word), by = "token") %>%
  as.data.table()
message("Tokens split in ", (proc.time()-timer)["elapsed"], " seconds.")

remove(all.news)

# Introduce lexicons and join to token data
# Note that inner joins will inherently filter out tokens
# not present in the given lexicon
message("joining dataset to sentiments")

# First the Afinn lexicon
message(". . . Afinn")
timer <- proc.time()
afinn <- fread("./data/afinn.csv")

# Change names of columns for joining
setnames(afinn, c("token", "sentiment"))

# setting data.table keys make joins a lot faster
setkey(afinn, token)
setkey(tokenized, token)

# data.table syntax for doing an inner join
afinn <- afinn[tokenized, nomatch = NULL]

# aggregate total sentiment for each article
afinn <- afinn[
  ,
  list(sentiment = sum(sentiment)),
  by  = list(title, is_fake, title_caps, set)
]

message("Afinn sentiment aggregated in ", (proc.time()-timer)["elapsed"], " seconds.")

# Join tokens to the NRC lexicon
message(". . . NRC")
timer <- proc.time()
nrc <- fread("./data/nrc.csv")

# Change column names for joining
setnames(nrc, "word", "token")

# Set keys for data.table join
setkey(nrc, token)
setkey(tokenized, token)

# data.table syntax for doing an inner join
nrc <- nrc[tokenized, nomatch = NULL]

# Tibbles with pivot_wider is a much easier-to-read approach here,
# but there are other more performant ways of doing this if our
# dataset was very large. See `reshape2` package
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

message("NRC sentiment aggregated in ", (proc.time()-timer)["elapsed"], " seconds.")

# Introduce the NRC VAD lexicon
message(". . . NRC VAD")
timer <- proc.time()
vad <- fread("./data/nrc_vad.csv")

# Change column names for joins
setnames(vad, tolower(names(vad)))
setnames(vad, "word", "token")

# set key for data.table join
setkey(vad, token)
setkey(tokenized, token)

# data.table syntax for an inner join
vad <- vad[tokenized, nomatch = NULL]

# Aggregate dimensions by summing across articles
vad <- vad[
  ,
  list(
    valence = sum(valence),
    arousal = sum(arousal),
    dominance = sum(dominance)
  ),
  by = list(title, is_fake, title_caps, set)
]
message("NRC VAD sentiment aggregated in ", (proc.time()-timer)["elapsed"], " seconds.")

## Splitting test and training sets for each of the lexicons
afinn <- split(afinn, by = "set", keep.by = FALSE)
afinn.training <- afinn$training
afinn.testing <- afinn$testing

nrc <- split(nrc, by = "set", keep.by = FALSE)
nrc.training <- nrc$training
nrc.testing <- nrc$testing

vad <- split(vad, by = "set", keep.by = FALSE)
vad.training <- vad$training
vad.testing <- vad$testing

# clean up
remove(tokenized, afinn, nrc, vad)

## Begin building models

# We can leveraging matrix-based function signatures for all the models we build
# This helper function will create a matrix of all predictors from a given data.table
makePredictors <- function(dt) {
  # drop our title, used only as an identifier column
  dt$title = NULL

  # drop the response column
  dt$is_fake = NULL

  # return the rest of the columns as a matrix
  as.matrix(dt)
}

## Random Forests
# Initialize and register parallel threads
cl <- makeSOCKcluster(nThreads)
registerDoSNOW(cl)

# Create a standard trainControl for all the RF models
# using 5-fold cross-validation in parallel
rf.trainControl <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE
)

# Random Forest model using the Afinn sentiment
message("RF Afinn")
timer <- proc.time()
afinn.rf.model <- train(
  makePredictors(afinn.training),
  afinn.training$is_fake,
  method = "parRF",
  trControl = rf.trainControl
)
message("Afinn Random Forest built in ", (proc.time()-timer)["elapsed"], " seconds.")

# Accuracy measures against training dataset
confusionMatrix(fitted(afinn.rf.model), afinn.training$is_fake)
# 82.3%

# Random Forest model using the NRC sentiments
message("RF NRC")
timer <- proc.time()
nrc.rf.model <- train(
  makePredictors(nrc.training),
  nrc.training$is_fake,
  method = "parRF",
  trControl = rf.trainControl
)
message("NRC Random Forest built in ", (proc.time()-timer)["elapsed"], " seconds.")

# Accuracy measures against training dataset
confusionMatrix(fitted(nrc.rf.model), nrc.training$is_fake)
# 99.7%

# Random Forest model using the NRC VAD dimensions
message("RF NRC VAD")
timer <- proc.time()
vad.rf.model <- train(
  makePredictors(vad.training),
  vad.training$is_fake,
  method = "parRF",
  trControl = rf.trainControl
)
message("NRC VAD Random Forest built in ", (proc.time()-timer)["elapsed"], " seconds.")

# Accuracy measures against training dataset
confusionMatrix(fitted(vad.rf.model), vad.training$is_fake)
# 99.8%

## Radial KSVMs
# create a matching trainControl for our KSVM models
ksvm.trainControl <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE
)

# KSVM model using the Afinn sentiment
message("KSVM Afinn")
timer <- proc.time()

# Create the matrix of predictors
afinn.training.predictors <- makePredictors(afinn.training)

# These KSVM Models are trained against a set of tuning parameters `sigma`,
# which effectively controls how linear or flexible the decision boundary
# becomes, and `C`, a measure of cost used to penalize large residuals after
# normalization. Since our dataset is small, no tested values for C < 1 had
# a positive effect, so for the sake of brevity it is defaulting to 1.

# Here, and below, sigmas are going to come from an estimation function provided
# in the `kernlab` package. The range of these will be expanded by 25% so that
# we can try a broader set of values for sigma.
afinn.training.sigmas <- sigest(afinn.training.predictors, frac = 1)
afinn.training.sigmas <- seq(
  afinn.training.sigmas["90%"] * 0.75,
  afinn.training.sigmas["10%"] * 1.25,
  length.out = 10
)

# train the model
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
message("Afinn SVM model built in ", (proc.time()-timer)["elapsed"], " seconds.")

# Accuracy measures against training dataset
confusionMatrix(fitted(afinn.ksvm.model$finalModel), afinn.training$is_fake)
# 82%

# KSVM model using the NRC sentiments
message("KSVM NRC")
timer <- proc.time()

# Create the matrix of predictors
nrc.training.predictors <- makePredictors(nrc.training)

# Create set of values for tuning sigma
nrc.training.sigmas <- sigest(nrc.training.predictors, frac = 1)
nrc.training.sigmas <- seq(
  nrc.training.sigmas["90%"] * 0.75,
  nrc.training.sigmas["10%"] * 1.25,
  length.out = 10
)

# Train the model
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
message("NRC SVM model built in ", (proc.time()-timer)["elapsed"], " seconds.")

# Accuracy measures against training dataset
confusionMatrix(fitted(nrc.ksvm.model$finalModel), nrc.training$is_fake)
# 89.3%

# KSVM model using the NRC VAD dimensions
message("KSVM VAD")
timer <- proc.time()

# Create the matrix of predictors
vad.training.predictors <- makePredictors(vad.training)

# Create the set of values for tuning sigma
vad.training.sigmas <- sigest(vad.training.predictors, frac = 1)
vad.training.sigmas <- seq(
  vad.training.sigmas["90%"] * 0.75,
  vad.training.sigmas["10%"] * 1.25,
  length.out = 10
)

# Train the model
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
message("NRC VAD SVM model built in ", (proc.time()-timer)["elapsed"], " seconds.")

# Accuracy measures against training dataset
confusionMatrix(fitted(vad.ksvm.model$finalModel), vad.training$is_fake)
# 86.1%

## Stop and deregister parallel computing
stopCluster(cl)
registerDoSEQ()
remove(cl)

# Build an ensemble model
# Create a container for our models
ensemble <- rbind(
  afinn.training[, list(title, is_fake)],
  nrc.training[, list(title, is_fake)],
  vad.training[, list(title, is_fake)]
)

ensemble <- unique(ensemble)

# create data.tables from the training data with a column for their respective
# predictions
afinn.rf <- cbind(
  afinn.training,
  afinn.rf = predict(afinn.rf.model, makePredictors(afinn.training))
)
nrc.rf <- cbind(
  nrc.training,
  nrc.rf = predict(nrc.rf.model, makePredictors(nrc.training))
)
vad.rf <- cbind(
  vad.training,
  vad.rf = predict(vad.rf.model, makePredictors(vad.training))
)
afinn.ksvm <- cbind(
  afinn.training,
  afinn.ksvm = predict(afinn.ksvm.model, makePredictors(afinn.training))
)
nrc.ksvm <- cbind(
  nrc.training,
  nrc.ksvm = predict(nrc.ksvm.model, makePredictors(nrc.training))
)
vad.ksvm <- cbind(
  vad.training,
  vad.ksvm = predict(vad.ksvm.model, makePredictors(vad.training))
)

# Remove columns not needed for this step
afinn.rf <- afinn.rf[, list(title, afinn.rf)]
nrc.rf <- nrc.rf[, list(title, nrc.rf)]
vad.rf <- vad.rf[, list(title, vad.rf)]
afinn.ksvm <- afinn.ksvm[, list(title, afinn.ksvm)]
nrc.ksvm <- nrc.ksvm[, list(title, nrc.ksvm)]
vad.ksvm <- vad.ksvm[, list(title, vad.ksvm)]

# Set keys
setkey(ensemble, title)
setkey(afinn.rf, title)
setkey(nrc.rf, title)
setkey(vad.rf, title)
setkey(afinn.ksvm, title)
setkey(nrc.ksvm, title)
setkey(vad.ksvm, title)

# a series of left-joins
ensemble <- afinn.rf[ensemble]
ensemble <- nrc.rf[ensemble]
ensemble <- vad.rf[ensemble]
ensemble <- afinn.ksvm[ensemble]
ensemble <- nrc.ksvm[ensemble]
ensemble <- vad.ksvm[ensemble]

# We can look at the matrix we've created
ensemble[, afinn.rf:vad.ksvm]

e <- rowMeans(do.call(cbind, lapply(ensemble[, afinn.rf:vad.ksvm], as.logical)), na.rm = TRUE)

e <- ensemble[
  e > 0.5,
  ensemble := "TRUE",
][
  e < 0.5,
  ensemble := "FALSE",
][
  e == 0.5,
  ensemble := sample(c("TRUE", "FALSE"), .N, replace = TRUE),
]

e[, ensemble := as.factor(ensemble)]

confusionMatrix(e$ensemble, e$is_fake)

# Final accuracy measures against testing set
# Afinn RF
confusionMatrix(predict(afinn.rf.model, makePredictors(afinn.testing)), afinn.testing$is_fake)
# 81.9

# NRC RF
confusionMatrix(predict(nrc.rf.model, makePredictors(nrc.testing)), nrc.testing$is_fake)
# 88.9

# NRC VAD RF
confusionMatrix(predict(vad.rf.model, makePredictors(vad.testing)), vad.testing$is_fake)
# 87.3


# Afinn SVM
confusionMatrix(predict(afinn.ksvm.model, makePredictors(afinn.testing)), afinn.testing$is_fake)
# 82.3

# NRC SVM
confusionMatrix(predict(nrc.ksvm.model, makePredictors(nrc.testing)), nrc.testing$is_fake)
# 87.3

# NRC VAD SVM
confusionMatrix(predict(vad.ksvm.model, makePredictors(vad.testing)), vad.testing$is_fake)
# 86.1

# clean up ensemble data
remove(e, ensemble, afinn.rf, nrc.rf, vad.rf, afinn.ksvm, nrc.ksvm, vad.ksvm)

# Ensemble model for testing dataset
# Build an ensemble model
# Create a container for our models
ensemble <- rbind(
  afinn.testing[, list(title, is_fake)],
  nrc.testing[, list(title, is_fake)],
  vad.testing[, list(title, is_fake)]
)

ensemble <- unique(ensemble)

# create data.tables from the testing data with a column for their respective
# predictions
afinn.rf <- cbind(
  afinn.testing,
  afinn.rf = predict(afinn.rf.model, makePredictors(afinn.testing))
)
nrc.rf <- cbind(
  nrc.testing,
  nrc.rf = predict(nrc.rf.model, makePredictors(nrc.testing))
)
vad.rf <- cbind(
  vad.testing,
  vad.rf = predict(vad.rf.model, makePredictors(vad.testing))
)
afinn.ksvm <- cbind(
  afinn.testing,
  afinn.ksvm = predict(afinn.ksvm.model, makePredictors(afinn.testing))
)
nrc.ksvm <- cbind(
  nrc.testing,
  nrc.ksvm = predict(nrc.ksvm.model, makePredictors(nrc.testing))
)
vad.ksvm <- cbind(
  vad.testing,
  vad.ksvm = predict(vad.ksvm.model, makePredictors(vad.testing))
)

# Remove columns not needed for this step
afinn.rf <- afinn.rf[, list(title, afinn.rf)]
nrc.rf <- nrc.rf[, list(title, nrc.rf)]
vad.rf <- vad.rf[, list(title, vad.rf)]
afinn.ksvm <- afinn.ksvm[, list(title, afinn.ksvm)]
nrc.ksvm <- nrc.ksvm[, list(title, nrc.ksvm)]
vad.ksvm <- vad.ksvm[, list(title, vad.ksvm)]

# Set keys
setkey(ensemble, title)
setkey(afinn.rf, title)
setkey(nrc.rf, title)
setkey(vad.rf, title)
setkey(afinn.ksvm, title)
setkey(nrc.ksvm, title)
setkey(vad.ksvm, title)

# a series of left-joins
ensemble <- afinn.rf[ensemble]
ensemble <- nrc.rf[ensemble]
ensemble <- vad.rf[ensemble]
ensemble <- afinn.ksvm[ensemble]
ensemble <- nrc.ksvm[ensemble]
ensemble <- vad.ksvm[ensemble]

# We can look at the matrix we've created
ensemble[, afinn.rf:vad.ksvm]

e <- rowMeans(do.call(cbind, lapply(ensemble[, afinn.rf:vad.ksvm], as.logical)), na.rm = TRUE)

e <- ensemble[
  e > 0.5,
  ensemble := "TRUE",
][
  e < 0.5,
  ensemble := "FALSE",
][
  e == 0.5,
  ensemble := sample(c("TRUE", "FALSE"), .N, replace = TRUE),
]

e[, ensemble := as.factor(ensemble)]

confusionMatrix(e$ensemble, e$is_fake)
#  86.8
