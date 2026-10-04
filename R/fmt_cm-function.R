fmt_cm <- function(x, lb, ub) ifelse(is.nan(x) | is.na(x), "—", sprintf("%.1f%% (%.1f, %.1f)", x, lb, ub))
