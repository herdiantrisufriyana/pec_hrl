clist_OD_uni_reg2=function(voi,ooi){
  reg_mod=
    paste0(ooi,'~',voi,'+grouping') %>%
    as.formula()
  
  suppressWarnings(
    lm(
        formula=reg_mod
        ,data=clist_OD_pheno_exprs
        ,weights=weight
      ) %>%
      tidy() %>%
      filter(term==voi)
  )
}
