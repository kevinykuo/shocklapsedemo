library(DALEX)
source("analysis/neural-net.R")
source("analysis/glm.R")
source("analysis/averages-method.R")
source("analysis/xgb.R")

predictors <- c(
  "gender", "issue_age", "face_amount", "post_level_premium_structure",
  "premium_jump_ratio", "risk_class", "premium_mode"
)

custom_predict_nn <- function(model, newdata) {
  keras_data <- make_keras_data(newdata, rec_nn)
  predict(model, keras_data$x)[[1]] %>%
    as.vector()
}

custom_predict_xgb <- function(model, newdata) {
  data <- bake(rec_xgb, newdata)
  xgb_data <-   xgb.DMatrix(
    data = Matrix::sparse.model.matrix(
      ~ . - 1,
      data = select(data, predictors)
    )
  )
  predict(model, xgb_data) %>%
    as.vector()
}

explainer_nn <- explain(
  model_nn, data = select(validation, predictors),
  y = validation$lapse_count_rate,
  predict_function = custom_predict_nn,
  label = "keras"
)

explainer_xgb <- DALEX::explain(
  model_xgb,
  data = select(validation, predictors),
  y = validation$lapse_count_rate,
  predict_function = custom_predict_xgb,
  label = "xgboost"
)

explainer_glm <- DALEX::explain(
  model_glm,
  data = bake(rec_glm, select(validation, predictors)),
  y = validation$lapse_count_rate,
  label = "glm"
)


newdata <- validation[7843,] %>% select(predictors)

# Prediction breakdown

pb_xgb <- prediction_breakdown(explainer_xgb, observation = newdata)
pb_nn <- prediction_breakdown(explainer_nn, observation = newdata)
pb_glm <- prediction_breakdown(explainer_glm, observation = newdata)
plot(pb_nn)
plot(pb_xgb)
plot(pb_glm)
plot(pb_xgb, pb_nn, pb_glm)

# Model performance

mp_nn <- model_performance(explainer_nn)
mp_xgb_model <- model_performance(explainer_xgb)
mp_glm <- model_performance(explainer_glm)
plot(mp_xgb_model, mp_glm, mp_nn)

# Variable importance

vi_nn <- variable_importance(explainer_nn, type = "ratio", n_sample = -1)
vi_xgb <- variable_importance(explainer_xgb, type = "ratio", n_sample = -1)
vi_glm <- variable_importance(explainer_glm, type = "ratio", n_sample = -1)

plot(vi_xgb, vi_glm, vi_nn)
