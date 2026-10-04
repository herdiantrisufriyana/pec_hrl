clist_OD_uni_reg=function(voi,ooi){
  reg_mod=
    paste0(ooi,'~',voi,'+grouping') %>%
    as.formula()
  
  suppressWarnings(
    glm(
        formula=reg_mod
        ,data=clist_OD_pheno_exprs
        ,family=binomial()
        ,weights=weight
      ) %>%
      tidy() %>%
      filter(term==voi)
  )
}
