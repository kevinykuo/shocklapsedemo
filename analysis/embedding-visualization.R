weights <- model_nn$get_layer("premium_jump_embedding")$get_weights()[[1]]

prcomp(weights, scale. = TRUE, rank = 2)$x[, c("PC1", "PC2")] %>%
  as.data.frame() %>%
  mutate(class = rec_nn$steps[[1]]$levels$premium_jump_ratio) %>%
  ggplot(aes(x = PC1, y = PC2)) +
  geom_point() +
  ggrepel::geom_text_repel(aes(label = class))
