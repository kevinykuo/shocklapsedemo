weights <- model_nn$get_layer("premium_jump_embedding")$get_weights()[[1]]

prcomp(weights, scale. = TRUE, rank = 2)$x[, c("PC1", "PC2")] %>%
  as.data.frame() %>%
  mutate(class = rec_nn$steps[[1]]$levels$premium_jump_ratio,
         ubound = purrr::map_dbl(
           class,
           ~ stringr::str_extract(.x, "((\\d|\\.)+)(?!.*\\d)") %>%
             as.numeric()
         ),
         group = case_when(
           ubound <= 10 ~ "<= 10",
           ubound <= 20 ~ "<= 20",
           ubound > 20 ~ "> 20"
         ) %>%
           as.factor()
  ) %>%
  ggplot(aes(x = PC1, y = PC2)) +
  geom_point() +
  ggrepel::geom_label_repel(aes(label = class, color = group))
