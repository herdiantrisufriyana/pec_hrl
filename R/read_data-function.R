read_data <- function(path){
  ext <- str_extract_all(path, "\\.[:alpha:]+$")[[1]]
  
  if(ext == ".csv") read_data_fn <- \(x) read_csv(x, show_col_types = FALSE)
  else if(ext == ".xlsx") read_data_fn <- \(x) readxl::read_xlsx(x, sheet = 1)
  else if(ext == ".sav") read_data_fn <- \(x) haven::read_sav(x)
  else if(ext == ".dta") read_data_fn <- \(x) haven::read_dta(x)
  else read_data_fn <- \(x) x
  
  read_data_fn(path)
}