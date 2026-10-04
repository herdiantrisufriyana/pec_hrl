compute_gradcam_eligible <- function(model_prefix) {
  gradcam_file <- paste0("data/gradcam/", model_prefix, ".csv")
  tidyset_file <- paste0("data/tidyset/", model_prefix, "_target.ts.tar.gz")

  if (!file.exists(gradcam_file) || !file.exists(tidyset_file)) return(NULL)

  # Read grad-CAM heatmap and add +1 to match ontotype 1-indexing (pec.Rmd line 10961)
  gc <- read.csv(gradcam_file, header = TRUE)
  colnames(gc) <- c("ontology", "x", "y", "z", "value")
  gc$x <- gc$x + 1
  gc$y <- gc$y + 1
  gc$z <- gc$z + 1

  # Clean up any previous extraction
  extract_dir <- sub("\\.ts\\.tar\\.gz$", "", tidyset_file)
  if (dir.exists(extract_dir)) unlink(extract_dir, recursive = TRUE)

  # Read TidySet and extract ontotype
  ts <- TidySet.read(tidyset_file)
  ontotype <- notes(experimentData(ts))[["ontotype"]]

  # Clean up extracted directory
  if (dir.exists(extract_dir)) unlink(extract_dir, recursive = TRUE)

  # Build gene → ontology → coordinate mapping from ontotype
  gene_gc_list <- lapply(names(ontotype), function(ont_id) {
    ot <- ontotype[[ont_id]]
    if (is.null(ot) || nrow(ot) == 0) return(NULL)
    df <- as.data.frame(ot)
    df$feature <- rownames(ot)
    df$ontology <- ont_id
    df
  })
  gene_coords <- do.call(rbind, gene_gc_list)
  if (is.null(gene_coords) || nrow(gene_coords) == 0) return(NULL)

  # Join with grad-CAM values
  gene_gc <- merge(gene_coords, gc, by = c("ontology", "x", "y", "z"), all.x = TRUE)
  gene_gc$value[is.na(gene_gc$value)] <- 0

  # Filter to value > 0 (matching pec.Rmd line 11093)
  gene_gc <- gene_gc[gene_gc$value > 0, ]

  eligible <- list()
  for (ont_id in unique(gene_gc$ontology)) {
    sub <- gene_gc[gene_gc$ontology == ont_id, ]
    if (nrow(sub) == 0) next
    q975 <- quantile(sub$value, 0.975)
    genes_above <- sub$feature[sub$value > q975]
    if (length(genes_above) > 0) {
      eligible[[ont_id]] <- as.character(genes_above)
    }
  }

  eligible
}
