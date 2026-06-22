prep_attribute_list <- function(data, attribute){
  if(attribute == "colname"){
    data.frame(old_colname = colnames(data), new_colname = NA)
  }else if(attribute == "coltype"){
    data.frame(
        old_coltype = sapply(data, \(x) paste0(class(x), collapse = "|"))
        , new_coltype = NA
        , row.names = colnames(data)
      ) |>
      rownames_to_column(var = "colname")
  }
}