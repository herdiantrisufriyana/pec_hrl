show_table <- function(data, name, caption, max_height = NULL, note = NULL){
  table_n <-
    read_csv("inst/extdata/list_of_tables.csv", show_col_types = FALSE) |>
    mutate(n = seq(n())) |>
    filter(variable == name) |>
    pull(n)
  
  table <-
    data |>
    kable(caption = paste0("Table ", table_n, ". ", caption), format = "html")
  
  if(!is.null(max_height)){
    table <-
      table |>
      kable_styling(full_width = TRUE) |>
      scroll_box(height = max_height)
  }
  
  if(!is.null(note)){
    table <- table |> footnote(note)
  }
  
  table |>
    kable_classic() |>
    column_spec(seq(ncol(data)), extra_css = "vertical-align:top;")
}