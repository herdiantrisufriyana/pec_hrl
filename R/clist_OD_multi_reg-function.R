clist_OD_multi_reg=function(data,voi,ooi,adjustment){
  reg_mod=
    paste0(ooi,'~',voi,adjustment) %>%
    as.formula()
  
  suppressWarnings(
    glm(
        formula=reg_mod
        ,data=data
        ,family=binomial()
        ,weights=weight
      ) %>%
      tidy() %>%
      filter(term==voi)
  )
}
