library(tidyverse)

data <- insurance::lapse_study %>%
  # Keep only duration 10 data.
  filter(duration == "10") %>%
  # Remove empty exposures.
  filter(exposure_count > 0, exposure_amount > 0) %>%
  mutate(
    lapse_count_rate = lapse_count / exposure_count,
    lapse_amount_rate = lapse_amount / exposure_amount
  )

training <- filter(data, policy_year < 2010)
validation <- filter(data, policy_year == 2010)
testing <- filter(data, policy_year >= 2011)

usethis::use_data(training, overwrite = TRUE)
usethis::use_data(validation, overwrite = TRUE)
usethis::use_data(testing, overwrite = TRUE)
