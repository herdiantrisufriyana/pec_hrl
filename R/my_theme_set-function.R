my_theme_set <- function() {
  ggplot2::theme_set(my_theme())
  
  cb_palette <- c(
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999"
  )
  
  options(
    ggplot2.discrete.color = cb_palette
    , ggplot2.discrete.fill = cb_palette
  )
}