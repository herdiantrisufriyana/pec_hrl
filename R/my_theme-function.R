my_theme <- function() {
  theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.text = element_text(color = "gray20"),
      panel.grid.major = element_line(color = "gray80"),
      panel.grid.minor = element_blank()
    )
}