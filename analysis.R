library(caret)
library(tidytext)
library(tidyr)
library(data.table)
library(dplyr)
library(dtplyr)
library(stringr)
library(kernlab)
library(doSNOW)

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
    text = str_trim(iconv(text, from = "utf8", to = "latin1")),
    is_cap_title = grepl('\\b?[A-Z]+\\b?', title)
  ) %>%
  filter(text != '' && title != '') %>%
  mutate(
    full_text = paste(text, title),
    is_fake = as.factor(is_fake),
    cap_title = as.factor(is_cap_title)
  ) %>%
  select(title, full_text, is_fake, is_cap_title) %>%
  as.data.table()

# Mark training and test datasets.
train.index <- createDataPartition(
  all.news$is_fake,
  p = 0.8,
  times = 1,
  list = FALSE
)

all.news$set <- "testing"
all.news[train.index, set := "training"]

remove(train.index)

# This takes a few moments
tokenize.timer <- proc.time()
# split tokens for joining sentiment, remove stop words.
tokenized <- all.news %>%
  unnest_tokens(token, full_text) %>%
  lazy_dt() %>%
  anti_join(data.table(token = stop_words$word), by = "token") %>%
  as.data.table()

tokenize.timer <- tokenize.timer - proc.time()

remove(all.news)


message("joining dataset to sentiments")

## Note that inner joins will inherently filter out tokens
## not present in the given lexicon

message(". . . Afinn")
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
  by  = list(title, is_fake, is_cap_title, set)
]

message(". . . NRC")
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
  group_by(title, is_fake, is_cap_title, set) %>%
  summarize_at(vars(-token), list(sum)) %>%
  as.data.table()


message(". . . NRC VAD")
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
  by = list(title, is_fake, is_cap_title, set)
]

## Remove title from all sets, they're not needed anymore
## and were only used as identifiers at this point
afinn$title = NULL
nrc$title = NULL
vad$title = NULL

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

remove(afinn, nrc, vad)


## Begin building models
## Random Forests
library(randomForest)
library(RRF)

cl <- makeSOCKcluster(nThreads)
registerDoSNOW(cl)

# RF AFINN
afinn.rf.model.timer <- proc.time()
afinn.rf.model <- train(
  is_fake ~ .,
  data = afinn.training,
  method = "RRF"
)
timer <- proc.time() - afinn.rf.model.timer

# RF NRC
nrc.rf.model.timer <- proc.time()
nrc.rf.model <- train(
  is_fake ~ .,
  data = nrc.training,
  method = "RRF"
)
timer <- proc.time() - nrc.rf.model.timer

# RF VAD
vad.rf.model.timer <- proc.time()
vad.rf.model <- train(
  is_fake ~ .,
  data = vad.training,
  method = "RRF"
)
timer <- proc.time() - vad.rf.model.timer

stopCluster(cl)
registerDoSEQ()
remove(cl)

## Radial KSVMs
# KSVM AFINN
# KSVM NRC
# KSVM VAD
