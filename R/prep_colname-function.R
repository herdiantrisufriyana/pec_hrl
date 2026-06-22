prep_colname <- function(data, colname_data){
  data |>
    `colnames<-`(
      data.frame(old_colname = colnames(data)) |>
        left_join(colname_data, by = join_by(old_colname)) |>
        pull(new_colname)
    )
}