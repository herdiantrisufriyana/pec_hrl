extract_thr_for_model <- function(prefix, eval_dir, ont_name_file = NULL) {
  all_thr <- list.files(eval_dir, pattern = paste0("^", prefix, "_(root|ONT[0-9]+)_thresholding\\.rds$"), full.names = TRUE)

  lapply(all_thr, function(f) {
    node <- str_extract(basename(f), "root|ONT[0-9]+")
    thr <- readRDS(f)
    dec_f <- sub("_thresholding\\.rds$", "_decision.rds", f)
    dec <- if (file.exists(dec_f)) readRDS(dec_f) else NULL

    original_id <- node
    if (!is.null(ont_name_file) && file.exists(ont_name_file) && node != "root") {
      ont_names <- read.csv(ont_name_file)
      ont_colon <- str_replace(node, "ONT", "ONT:")
      m <- ont_names[ont_names$new_oid == ont_colon, ]
      if (nrow(m) > 0) original_id <- m$old_oid[1]
    }

    threshold <- thr$ref_value[1]
    nb_model <- thr$avg[thr$metric == "nb"]
    nb_ref <- NA
    if (!is.null(dec) && length(dec$plot$layers) >= 3) {
      ta <- dec$plot$layers[[3]]$data
      nb_treat_all <- ta$intercept + ta$slope * threshold
      nb_ref <- max(nb_treat_all, 0)
    }

    list(node = node, original_id = original_id, thr = thr,
         threshold = threshold, nb_model = nb_model, nb_ref = nb_ref)
  })
}
