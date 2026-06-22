prep_coltype <- function(data, coltype_data, remove_haven_label = FALSE){
  if(remove_haven_label){
    data |>
      mutate_if(
        data |>
          sapply(
            \(x) str_detect(paste0(class(x), collapse = "|"), "haven_labelled")
          )
        , \(x) haven::as_factor(x)
      )
  }else{
    class_converter <-
      list(
        \(x) as_date(x)
        , \(x) as.factor(x)
        , \(x) as.numeric(x)
        , \(x) as.character(x)
      )
    
    new_coltype <-
      data.frame(colname = colnames(data)) |>
      left_join(coltype_data, by = join_by(colname)) |>
      pull(new_coltype) |>
      lapply(
        \(x)
        case_when(
          x == "date" ~ 1
          , x == "factor" | x == "factor_numeric" ~ 2
          , x == "numeric" ~ 3
          , TRUE ~ 4
        )
      ) |>
      `names<-`(colnames(data))
    
    data |>
      imap(~ class_converter[[new_coltype[[.y]]]](.x)) |>
      lapply(as.data.frame) |>
      imap(~ `colnames<-`(.x, .y)) |>
      reduce(cbind)
  }
}