fmt_pair <- function(a_avg, a_lb, a_ub, b_avg, b_lb, b_ub) {
  paste0(fmt_cm(a_avg, a_lb, a_ub), " / ", fmt_cm(b_avg, b_lb, b_ub))
}
