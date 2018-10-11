library(shocklapsedemo)
library(tidyverse)
library(recipes)
library(keras)
library(yardstick)

predictors <- c(
  "gender", "issue_age", "face_amount", "post_level_premium_structure",
  "premium_jump_ratio", "risk_class", "premium_mode"
)
responses <- c("lapse_count_rate", "lapse_amount_rate")

f <- as.formula(paste(
  paste(responses, collapse = " + "),
  paste(predictors, collapse = " + "),
  sep = "~"
))

rec_nn <- recipe(f, data = training) %>%
  step_string2factor(all_predictors()) %>%
  prep(retain = TRUE, stringsAsFactors = FALSE)

use_session_with_seed(2018)

training_data <- make_keras_data(training, rec_nn)
testing_data <- make_keras_data(testing, rec_nn)
validation_data <- make_keras_data(validation, rec_nn)

model_nn <- keras_model_lapse()
model_nn %>%
  compile(
    optimizer = optimizer_adam(amsgrad = TRUE),
    loss = "mse",
    loss_weights = c(0.8, 0.2)
  )

history <- model_nn %>%
  fit(
    x = training_data$x,
    y = training_data$y,
    batch_size = 256,
    epochs = 100
  )

predictions_nn <- bind_cols(
  validation,
  predict(model_nn, validation_data$x) %>%
    setNames(c("predicted_count_rate", "predicted_amount_rate")) %>%
    as.data.frame()
)

metrics(predictions_nn, lapse_count_rate, predicted_count_rate)

predictions_nn %>%
  compute_prediction_quantiles("predicted_count_rate", "lapse_count_rate") %>%
  plot_actual_vs_expected()

# Evaluate on testing set

full_training_data <- make_keras_data(bind_rows(training, validation), rec_nn)

model_nn <- keras_model_lapse()
model_nn %>%
  compile(
    optimizer = optimizer_adam(amsgrad = TRUE),
    loss = "mse",
    loss_weights = c(0.8, 0.2)
  )

history <- model_nn %>%
  fit(
    x = full_training_data$x,
    y = full_training_data$y,
    batch_size = 256,
    epochs = 50
  )

predictions_nn <- bind_cols(
  testing,
  predict(model_nn, testing_data$x) %>%
    setNames(c("predicted_count_rate", "predicted_amount_rate")) %>%
    as.data.frame()
)

metrics(predictions_nn, lapse_count_rate, predicted_count_rate)

predictions_nn %>%
  compute_prediction_quantiles("predicted_count_rate", "lapse_count_rate") %>%
  plot_actual_vs_expected()
