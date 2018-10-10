
#' Compute Average Predictions by Quantile
#'
#' Compute the average predicted and actual values by decile
#'
#' @param predictions Data frame of predictions.
#' @param predicted_col Column name of predicted values.
#' @param actual_col Column name of actual values.
#'
#' @importFrom dplyr .data
#' @export
compute_prediction_quantiles <- function(predictions, predicted_col, actual_col) {
  predictions %>%
    dplyr::select(dplyr::one_of(c(predicted_col, actual_col))) %>%
    dplyr::mutate(decile = cut(!!rlang::sym(predicted_col), unique(stats::quantile(
      !!rlang::sym(predicted_col), probs = seq(0, 1, 0.1)
    )), include.lowest = TRUE)) %>%
    dplyr::group_by(.data$decile) %>%
    dplyr::summarize(mean_predicted = mean(!!rlang::sym(predicted_col)),
                     mean_actual = mean(!!rlang::sym(actual_col)))
}

#' Plot Actual vs. Predicted
#'
#' @param df Data frame of predictions and actuals by quantile. Should be the
#'   return value of `compute_prediction_quantiles()`.
#' @export
plot_actual_vs_expected <- function(df) {
  df %>%
    tidyr::gather("key", "value", -"decile") %>%
    ggplot2::ggplot(ggplot2::aes(x = "decile", y = "value", fill = "key")) +
    ggplot2::geom_bar(stat = "identity", position = "dodge") +
    ggplot2::ggtitle("Average Actual vs. Predicted Lapse Rates") +
    ggplot2::xlab("Actual Lapse Rate Decile") +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
    )
}

#' Initialize Keras Model for Lapse Prediction
#'
#' @import keras
#' @export
keras_model_lapse <- function() {
  input_gender <- layer_input(shape = 2, name = "gender")
  input_issue_age_group <- layer_input(shape = 1, name = "issue_age")
  input_face_amount_band <- layer_input(shape = 1, name = "face_amount")
  input_post_level_premium_structure <- layer_input(shape = 2, name = "post_level_premium_structure")
  input_prem_jump_d11_d10 <- layer_input(shape = 1, name = "premium_jump_ratio")
  input_risk_class <- layer_input(shape = 1, name = "risk_class")
  input_premium_mode <- layer_input(shape = 6, name = "premium_mode")

  concat_inputs <- layer_concatenate(list(
    input_gender,
    input_issue_age_group %>%
      layer_embedding(7, 2) %>%
      layer_flatten(),
    input_face_amount_band %>%
      layer_embedding(4, 2) %>%
      layer_flatten(),
    input_post_level_premium_structure,
    input_prem_jump_d11_d10 %>%
      layer_embedding(25, 24) %>%
      layer_flatten(),
    input_risk_class %>%
      layer_embedding(9, 2) %>%
      layer_flatten(),
    input_premium_mode
  ))

  main_layer <- concat_inputs %>%
    layer_dense(units = 64, activation = "relu")

  output_count_rate <- main_layer %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid", name = "lapse_count_rate")

  output_amount_rate <- main_layer %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid", name = "lapse_amount_rate")

  model <- keras_model(
    inputs = c(input_gender, input_issue_age_group, input_face_amount_band,
               input_post_level_premium_structure, input_prem_jump_d11_d10,
               input_risk_class, input_premium_mode),
    outputs = c(output_count_rate, output_amount_rate)
  )
}

#' Preprocess Data for Keras Lapse Model
#'
#' @param data Dataset with categorical variables as factors.
#' @param recipe Trained recipe object.
#' @export
make_keras_data <- function(data, recipe) {
  data <- recipes::bake(recipe, data) %>%
    purrr::map_if(is.factor, ~ as.integer(.x) - 1) %>%
    purrr::map_at("gender", ~ keras::to_categorical(.x, 2) %>% array_reshape(c(length(.x), 2))) %>%
    purrr::map_at("post_level_premium_structure",
                  ~ keras::to_categorical(.x, 2) %>% array_reshape(c(length(.x), 2))) %>%
    purrr::map_at("premium_mode", ~ keras::to_categorical(.x, 6) %>%
                    array_reshape(c(length(.x), 6)))

  predictors <- recipe$var_info$variable[recipe$var_info$role == "predictor"]
  responses <- recipe$var_info$variable[recipe$var_info$role == "outcome"]

  list(x = data[predictors],
       y = data[responses])
}

#' Preprocess data for XGB model
#'
#' @param data Dataset with categorical variables as factors.
#' @param recipe Trained recipe object.
#' @export
make_xgb_data <- function(data, recipe) {
  data <- recipes::bake(recipe, data)
  predictors <- recipe$var_info$variable[recipe$var_info$role == "predictor"]
  xgboost::xgb.DMatrix(
    data = Matrix::sparse.model.matrix(
      ~ . - 1,
      data = dplyr::select(data, predictors)
    ),
    label = data$lapse_count_rate
  )
}
