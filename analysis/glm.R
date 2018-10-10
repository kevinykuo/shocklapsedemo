library(yardstick)
library(tidyverse)
library(recipes)
f <- lapse_count_rate ~ gender + issue_age + face_amount + post_level_premium_structure +
  premium_jump_ratio + risk_class + premium_mode
rec_glm <- recipe(f, data = training) %>%
  step_string2factor(all_predictors()) %>%
  prep(retain = TRUE, stringsAsFactors = FALSE)
model_glm <- glm(
  f,
  family = gaussian,
  data = juice(rec_glm)
)

predictions_glm <- bind_cols(
  validation,
  data.frame(predicted_count_rate = predict(model_glm, validation))
)
metrics(predictions_glm, lapse_count_rate, predicted_count_rate)

# Evaluate on testing

model_glm <- glm(
  f,
  family = gaussian,
  data = bake(rec_glm, bind_rows(training, validation))
)

predictions_glm <- bind_cols(
  testing,
  data.frame(predicted_count_rate = predict(model_glm, testing))
)
metrics(predictions_glm, lapse_count_rate, predicted_count_rate)
