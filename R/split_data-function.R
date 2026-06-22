split_data <-
  function(
    data
    , seed
    , name = c("partition1", "partition2")
    , partition1_prop = 0.2
    , identifier = NULL
    , strata = NULL
  ){
    if(is.null(identifier)){
      whole_set <-
        data |>
        mutate(identifier = seq(nrow(data))) |>
        select(identifier, everything())
    }else{
      whole_set <-
        data |>
        select_at(identifier) |>
        rename_at(identifier, \(x) "identifier") |>
        cbind(data)
    }
    
    if(is.null(strata)){
      whole_set <-
        whole_set |>
        mutate(strata = "strata")
    }else{
      whole_set <-
        whole_set |>
        select_at(strata) |>
        rename_at(strata, \(x) "strata") |>
        cbind(whole_set)
    }
    
    whole_set <-
      whole_set |>
      select(identifier, strata) |>
      unique()
    
    whole_id <-
      whole_set |>
      pull(strata) |>
      unique() |>
      sort() |>
      lapply(\(x) filter(whole_set, strata == x)$identifier)
    
    partition1 <- list()
    partition2 <- list()
    
    for(i in seq(length(whole_id))){
      set.seed(seed)
      
      partition1[[i]] <-
        whole_id[[i]] |>
        sample(size = round(0.2 * length(whole_id[[i]])), replace = FALSE)
      
      partition2[[i]] <- setdiff(whole_id[[i]], partition1[[i]])
    }
    
    whole_id <- reduce(whole_id, c)
    partition1 <- reduce(partition1, c)
    partition2 <- reduce(partition2, c)
    
    partition <- setNames(list(partition1, partition2), name)
    
    return(partition)
  }