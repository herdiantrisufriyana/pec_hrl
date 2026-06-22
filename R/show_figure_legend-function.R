show_figure_legend <- function(name, caption, note = NULL){
  figure_n <-
    read_csv("inst/extdata/list_of_figures.csv", show_col_types = FALSE) |>
    mutate(n = seq(n())) |>
    filter(variable == name) |>
    pull(n)
  
  cat(
    paste0(
      "Figure ", figure_n,". "
      , caption
      , ifelse(is.null(note), "", " ")
      , paste0(note, collapse = "")
      , ifelse(is.null(note), "", ".")
    )
  )
}