extract_metrics_for_model <- function(prefix, eval_dir, ont_name_file = NULL) {
  # Get all discrimination files (root + ontology)
  all_disc <- list.files(eval_dir, pattern = paste0("^", prefix, "_(root|ONT[0-9]+)_discrimination\\.rds$"), full.names = TRUE)

  results <- lapply(all_disc, function(f) {
    node <- str_extract(basename(f), "root|ONT[0-9]+")
    disc <- readRDS(f)
    cal_f <- sub("_discrimination\\.rds$", "_calibration.rds", f)
    cal <- if (file.exists(cal_f)) readRDS(cal_f) else NULL

    auroc <- disc$metrics$estimate
    auroc_ci <- disc$metrics$ci
    auprc <- if (!is.null(cal)) tryCatch(compute_auprc(disc, cal), error = function(e) NA) else NA
    rmse <- if (!is.null(cal)) cal$metrics$estimate[3] else NA
    rmse_ci <- if (!is.null(cal)) cal$metrics$ci[3] else NA

    # Map ONT ID to original
    original_id <- node
    if (!is.null(ont_name_file) && file.exists(ont_name_file) && node != "root") {
      ont_names <- read.csv(ont_name_file)
      ont_colon <- str_replace(node, "ONT", "ONT:")
      m <- ont_names[ont_names$new_oid == ont_colon, ]
      if (nrow(m) > 0) original_id <- m$old_oid[1]
    }

    data.frame(node = node, original_id = original_id,
               auroc = auroc, auroc_ci = auroc_ci,
               auprc = auprc, rmse = rmse, rmse_ci = rmse_ci)
  })

  do.call(rbind, results)
}
