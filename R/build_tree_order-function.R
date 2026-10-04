build_tree_order <- function(ontology_csv_path, ont_name_file = NULL) {
  raw <- read.csv(ontology_csv_path, header = FALSE,
                  col.names = c("source", "target", "similarity", "relation"))

  # Separate hierarchy (is_a) and features
  hierarchy <- raw[raw$relation == "is_a", c("source", "target")]
  features <- raw[raw$relation == "feature", c("source", "target")]

  # Find root: target that's never a source in hierarchy
  all_parents <- unique(hierarchy$target)
  all_children <- unique(hierarchy$source)
  root_node <- setdiff(all_parents, all_children)
  if (length(root_node) == 0) root_node <- all_parents[1]
  root_node <- root_node[1]

  # Get genes per ontology node
  genes_per_node <- split(features$source, features$target)

  # Map ONT IDs to original names
  ont_map <- setNames(nm = unique(c(hierarchy$source, hierarchy$target)))
  if (!is.null(ont_name_file) && file.exists(ont_name_file)) {
    ont_names <- read.csv(ont_name_file)
    for (j in seq_len(nrow(ont_names))) {
      ont_map[ont_names$new_oid[j]] <- ont_names$old_oid[j]
    }
  }

  # Build children map
  children_of <- split(hierarchy$source, hierarchy$target)

  # DFS traversal to get tree order with prefixes
  result <- list()
  dfs <- function(node, prefix, is_last) {
    mapped <- if (!is.null(ont_map[node]) && !is.na(ont_map[node])) ont_map[node] else node
    # Strip ONT: prefix for matching with eval files
    eval_node <- gsub(":", "", node)

    genes <- genes_per_node[[node]]
    genes_vec <- if (!is.null(genes) && length(genes) > 0) genes else character(0)
    gene_str <- if (length(genes_vec) > 0) paste0(" (", paste(genes_vec, collapse=", "), ")") else ""

    label <- paste0(mapped, gene_str)

    if (node == root_node) {
      display <- paste0(label, gene_str)
      eval_id <- eval_node
    } else {
      connector <- if (is_last) "\U2514\U2500\U2500 " else "\U251C\U2500\U2500 "
      display <- paste0(prefix, connector, label)
      eval_id <- eval_node
    }

    result[[length(result) + 1]] <<- list(display = display, eval_node = eval_id, genes = genes_vec)

    kids <- children_of[[node]]
    if (!is.null(kids) && length(kids) > 0) {
      kids <- sort(kids)
      for (k in seq_along(kids)) {
        child_is_last <- (k == length(kids))
        new_prefix <- if (node == root_node) {
          ""
        } else {
          paste0(prefix, if (is_last) "    " else "\U2502   ")
        }
        dfs(kids[k], new_prefix, child_is_last)
      }
    }
  }

  dfs(root_node, "", TRUE)
  data.frame(
    display = sapply(result, \(x) x$display),
    eval_node = sapply(result, \(x) x$eval_node),
    genes = I(lapply(result, \(x) x$genes)),
    stringsAsFactors = FALSE
  )
}
