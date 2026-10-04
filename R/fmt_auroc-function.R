fmt_auroc <- function(a, ci) sprintf("%.3f (%.3f, %.3f)", a, a - ci, a + ci)
