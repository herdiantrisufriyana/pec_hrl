clist_OD_adjustment_check=function(voi,ooi,covariates){
  # covariates=
  #   clist_OD_inter_genes %>%
  #   filter(abs(r)>=0.4 & p<=0.05) %>%
  #   select(-r,-p) %>%
  #   filter(Var1==voi|Var2==voi) %>%
  #   gather(variable,value) %>%
  #   filter(variable!=voi) %>%
  #   pull(value) %>%
  #   unique()
  
  if(length(covariates)>0){
    reg_mod=
      covariates %>%
      `names<-`(as.character(.)) %>%
      lapply(\(x)paste0(ooi,'~',voi,'+',x,'+grouping')) %>%
      lapply(as.formula)
    
    reg_mod %>%
      pblapply(X=names(.),Y=.,\(X,Y)
        suppressWarnings(
          glm(
              formula=Y[[X]]
              ,data=clist_OD_pheno_exprs
              ,family=binomial()
              ,weights=weight
            ) %>%
            tidy() %>%
            filter(term==voi) %>%
            mutate(covariate=X) %>%
            select(term,covariate,everything())
        )
      ) %>%
      purrr::reduce(rbind)
  }else{
    NULL
  }
}
