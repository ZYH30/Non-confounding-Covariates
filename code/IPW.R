dataframe <- read.table('../data/data-discrete.csv', header = TRUE, sep = ',')
colnames(dataframe)
ATE_True <- mean(dataframe$Y_1 - dataframe$Y_0)
ATE_Dire <- mean(dataframe[dataframe$T == 1,'Y']) - mean(dataframe[dataframe$T == 0,'Y'])

IPW_ATE_Effect <- function(PS, DataFrame){
  S_PS_1 <- sum(DataFrame$T / PS)
  S_PS_0 <- sum((1 - DataFrame$T) / (1-PS))
  
  E_t_1 <- sum(DataFrame$T * DataFrame$Y / PS / S_PS_1)
  E_t_0 <- sum((1 - DataFrame$T) * DataFrame$Y / (1-PS) / S_PS_0)
  ATE_IPW <- E_t_1 - E_t_0
  
  return(ATE_IPW)
}

library(nnet)
library(gbm)

IPW_ITE_Effect <- function(PS, DataFrame, X, model = 'tree'){
  PS_1 <- DataFrame$T / PS / sum(DataFrame$T / PS)
  PS_0 <- (1 - DataFrame$T) / (1-PS) / sum((1 - DataFrame$T) / (1-PS))

  # split
  DataFrame_1 <- DataFrame[DataFrame$T == 1,c('T',X,'Y','Y_0','Y_1')]
  PS_1_ <- PS_1[DataFrame$T == 1]
  
  DataFrame_0 <- DataFrame[DataFrame$T == 0,c('T',X,'Y','Y_0','Y_1')]
  PS_0_ <- PS_0[DataFrame$T == 0]
  
  if(model == 'tree'){
    form_X <- paste(X, collapse = "+")
    form <- formula(paste(c('Y',form_X), collapse = "~"))
    
    # t = 1
    model_lm_1 <- gbm(formula = form,data = DataFrame_1, distribution = "gaussian", weights = PS_1_,
                      n.trees = 100, shrinkage = 0.1,             
                      interaction.depth = 3, bag.fraction = 0.5, train.fraction = 0.6,  
                      n.minobsinnode = 10, cv.folds = 1, keep.data = TRUE, 
                      verbose = FALSE, n.cores = 1)
    best.iter_1 <- gbm.perf(model_lm_1, method = "test")
    # t = 0
    model_lm_0 <- gbm(formula = form,data = DataFrame_0, distribution = "gaussian", weights = PS_0_,
                      n.trees = 100, shrinkage = 0.1,             
                      interaction.depth = 3, bag.fraction = 0.5, train.fraction = 0.6,  
                      n.minobsinnode = 10, cv.folds = 1, keep.data = TRUE, 
                      verbose = FALSE, n.cores = 1)
    best.iter_0 <- gbm.perf(model_lm_0, method = "test")
  } else if(model == 'poly'){

    form <- paste0('Y~',paste0(paste0('poly(', X,sep = ","),'3)',collapse="+"))
    # t = 1
    model_lm_1 <- lm(formula = form,data = DataFrame_1,weights = PS_1_)
    # t = 0
    model_lm_0 <- lm(formula = form,data = DataFrame_0,weights = PS_0_)
  }
  
  
  # pehe
  if(model == 'tree'){
    predict_y_0 <- predict(model_lm_0,newdata = DataFrame_1,n.trees = best.iter_0)
  } else if(model == 'poly'){
    predict_y_0 <- predict(model_lm_0,newdata = DataFrame_1)
  }
  
  ITE_PRE_1 <- DataFrame_1$Y - predict_y_0
  ITE_true_1 <- DataFrame_1$Y_1 - DataFrame_1$Y_0
  PEHE_1_ <- sum((ITE_PRE_1 - ITE_true_1)^2) 
  # y1
  if(model == 'tree'){
    predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0,n.trees = best.iter_1)
  } else if(model == 'poly'){
    predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0)
  }
  
  ITE_PRE_0 <- predict_y_1 - DataFrame_0$Y
  ITE_true_0 <- DataFrame_0$Y_1 - DataFrame_0$Y_0
  PEHE_0_ <- sum((ITE_PRE_0 - ITE_true_0)^2) 
  
  # PEHE
  PEHE = (PEHE_1_ + PEHE_0_) / (length(ITE_PRE_1) + length(ITE_PRE_0))
  return(PEHE)
  
}
IPW_FUNC <- function(T, X, DataFrame){
  form_X <- paste(X, collapse = "+")
  form <- formula(paste(c(T,form_X), collapse = "~"))
  
  logit <- glm(form, family = binomial(link = 'logit'), DataFrame)
  PS <- logit$fitted.values
  
  ATE_IPW <- IPW_ATE_Effect(PS,DataFrame)
  PEHE <- IPW_ITE_Effect(PS,DataFrame,X)
  
  return(c(ATE_IPW,PEHE))
}

## C
R_IPW_C <- IPW_FUNC('T',c('C'),dataframe)
abs(R_IPW_C[1] - ATE_True)
R_IPW_C[2]

X1 = c( "I","A","YO","M","TO","Z")
for(x in X1){
  R_IPW_ <- IPW_FUNC('T',c('C',x),dataframe)
  ATE <- R_IPW_[1];PEHE = R_IPW_[2]
  
  ATE_e <- abs(ATE - ATE_True)
  print(paste(c('################',x,'################'), collapse = ""))
  print(paste(c('ATE_ERROR',ATE_e), collapse = ":"))
  print(paste(c('PEHE',PEHE), collapse = ":"))
}
