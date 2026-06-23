compute_metrics=
  function(
    eval_df
    ,p=T
    ,tpr=T
    ,tnr=T
    ,ppv=T
    ,npv=T
    ,nb=T
    ,interp=F
  ){
    cm=
      eval_df %>%
      cbind(
        seq(0,1,0.01) %>%
          matrix(
            nrow=1
            ,byrow=T
            ,dimnames=list(NULL,str_pad(.,str_count(max(.)),'left','0'))
          ) %>%
          as.data.frame()
      ) %>%
      mutate(seq=seq(nrow(.))) %>%
      select(seq,everything()) %>%
      gather(th,value,-seq,-pred,-obs) %>%
      mutate(
        pred=
          factor(
            ifelse(pred>=value,levels(obs)[2],levels(obs)[1])
            ,levels(obs)
          )
      ) %>%
      group_by(th) %>%
      summarize(
        tp=sum(obs==levels(obs)[2] & pred==levels(obs)[2])
        ,fn=sum(obs==levels(obs)[2] & pred==levels(obs)[1])
        ,fp=sum(obs==levels(obs)[1] & pred==levels(obs)[2])
        ,tn=sum(obs==levels(obs)[1] & pred==levels(obs)[1])
        ,.groups='drop'
      ) %>%
      mutate(th=as.numeric(th))
    
    metrics=cm
    
    if(p){
      metrics=
        metrics %>%
        mutate(p=(tp+fn)/(tp+fn+fp+tn))
    }
    
    if(tpr){
      metrics=
        metrics %>%
        mutate(tpr=tp/(tp+fn))
    }
    
    if(tnr){
      metrics=
        metrics %>%
        mutate(tnr=tn/(fp+tn))
    }
    
    if(ppv){
      metrics=
        metrics %>%
        mutate(ppv=tp/(tp+fp))
    }
    
    if(npv){
      metrics=
        metrics %>%
        mutate(npv=tn/(tn+fn))
    }
    
    if(nb){
      metrics=
        metrics %>%
        mutate(nb=(tp-fp*th/(1-th))/(tp+fn+fp+tn))
    }
    
    metrics=
      metrics %>%
      select(-tp,-fn,-fp,-tn) %>%
      mutate_all(round,4)
    
    if(interp){
      for(colname in colnames(metrics)){
        metrics=
          metrics %>%
          full_join(
            select_at(.,colname) %>%
              setNames('metric') %>%
              filter(!is.na(metric)) %>%
              filter(!is.nan(metric)) %>%
              filter(metric==min(metric)| metric==max(metric)) %>%
              filter(!duplicated(.)) %>%
              arrange(metric) %>%
              pull(metric) %>%
              lapply(X=1,Y=.,function(X,Y)seq(Y[1],Y[2],0.0001)) %>%
              .[[1]] %>%
              data.frame(metric=.) %>%
              setNames(colname)
            ,by=colname
          )
        
        for(colname2 in colnames(metrics) %>% .[.!=colname]){
          metrics=
            metrics %>%
            arrange_at(c(colname,colname2)) %>%
            mutate_at(
              colname2
              ,function(x) approx(seq(length(x)),x,seq(length(x)))$y
            )
        }
        
        metrics=
          metrics %>%
          mutate_all(round,4)
      }
    }
    
    list(
      cm=cm
      ,metrics=metrics
    )
  }