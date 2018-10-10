library(shocklapse)
library(tidyverse)
library(yardstick)
library(xgboost)
library(recipes)

predictors <- c(
  "gender", "issue_age", "face_amount", "post_level_premium_structure",
  "premium_jump_ratio", "risk_class", "premium_mode"
)

rec_xgb <- recipe(f, data = training) %>%
  step_string2factor(all_predictors()) %>%
  prep(retain = TRUE, stringsAsFactors = FALSE)

training_data <- make_xgb_data(training, rec_xgb)
validation_data <- make_xgb_data(validation, rec_xgb)
full_training_data <- make_xgb_data(bind_rows(training, validation), rec_xgb)
testing_data <- make_xgb_data(testing, rec_xgb)

param_list <- list(
  subsample = c(0.5, 0.7),
  max_depth = c(5, 10),
  eta = c(0.1, 0.2)
)

xgb_models <- param_list %>%
  purrr::cross() %>%
  purrr::map(xgb.cv, data = training_data, nrounds = 400, metrics = "rmse", nfold = 3,
             early_stopping_rounds = 10, objective = "reg:logistic")

cv_summary <- xgb_models %>%
  map_df(~ c(
    .x$params,
    best_iteration = .x$best_iteratio,
    rmse = .x$evaluation_log$test_rmse_mean %>% tail(1)
  ))

set.seed(2018)
model_xgb <- xgboost(data = training_data, params = list(
  subsample = 0.7,
  max_depth = 5,
  eta = 0.1), nrounds = 350)

prediction <- predict(model_xgb, validation_data)

predictions_xgb <- validation %>% mutate(
  predicted_count_rate = prediction
)

metrics(predictions_xgb, lapse_count_rate, predicted_count_rate)

# Evaluate on testing

model_xgb <- xgboost(data = full_training_data, params = list(
  subsample = 0.7,
  max_depth = 5,
  eta = 0.1), nrounds = 400)

prediction <- predict(model_xgb, testing_data)
predictions_xgb <- testing %>% mutate(
  predicted_count_rate = prediction
)

metrics(predictions_xgb, lapse_count_rate, predicted_count_rate)
