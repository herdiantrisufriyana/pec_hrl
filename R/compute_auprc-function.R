compute_auprc <- function(disc_rds, cal_rds) {
  roc <- disc_rds$plot$data
  d <- cal_rds$dist$data
  p <- d$n[d$obs == "1"] / sum(d$n)
  precision <- (roc$tpr * p) / (roc$tpr * p + (1 - roc$tnr) * (1 - p))
  recall <- roc$tpr
  precision[is.nan(precision)] <- 1
  ord <- order(recall, precision)
  recall <- recall[ord]
  precision <- precision[ord]
  idx <- !duplicated(recall)
  recall <- recall[idx]
  precision <- precision[idx]
  sum(diff(recall) * (precision[-1] + precision[-length(precision)]) / 2, na.rm = TRUE)
}
