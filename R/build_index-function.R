build_index <- function() {
  input <- "index.Rmd"
  name <- tools::file_path_sans_ext(basename(input))
  
  rmarkdown::render(
    input = input,
    output_file = paste0(name, ".html")
  )
  
  knitr::purl(
    input = input,
    output = paste0(name, ".R")
  )
  
  invisible(TRUE)
}