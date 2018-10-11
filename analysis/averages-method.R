library(shocklapsedemo)
library(tidyverse)
library(yardstick)
# no machine learning here, just taking historical averages
predictions_averages <- training %>%
  group_by(gender, issue_age, face_amount,
           post_level_premium_structure, premium_jump_ratio,
           risk_class, premium_mode) %>%
  summarize(predicted_count_rate = sum(lapse_count) / sum(exposure_count),
            predicted_amount_rate = sum(lapse_amount) / sum(exposure_amount)) %>%
  right_join(validation)

metrics(as.data.frame(predictions_averages), lapse_count_rate, predicted_count_rate)

# Evaluate on testing

predictions_averages <- bind_rows(training, validation) %>%
  group_by(gender, issue_age, face_amount,
           post_level_premium_structure, premium_jump_ratio,
           risk_class, premium_mode) %>%
  summarize(predicted_count_rate = sum(lapse_count) / sum(exposure_count),
            predicted_amount_rate = sum(lapse_amount) / sum(exposure_amount)) %>%
  right_join(testing)

metrics(as.data.frame(predictions_averages), lapse_count_rate, predicted_count_rate)
