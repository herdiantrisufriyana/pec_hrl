get_metric <- function(thr_rds, m) {
  list(avg = thr_rds$avg[thr_rds$metric == m],
       lb = thr_rds$lb[thr_rds$metric == m],
       ub = thr_rds$ub[thr_rds$metric == m])
}
