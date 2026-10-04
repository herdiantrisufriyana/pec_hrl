fmt_rmse <- function(r, ci) ifelse(is.na(r), "—", sprintf("%.3f (±%.4f)", r, ci))
