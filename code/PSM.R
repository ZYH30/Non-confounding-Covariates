dataframe <- read.table('../data/data-discrete.csv', header = TRUE, sep = ',')
colnames(dataframe)
ATE_True <- mean(dataframe$Y_1 - dataframe$Y_0)
ATE_Dire <- mean(dataframe[dataframe$T == 1,'Y']) - mean(dataframe[dataframe$T == 0,'Y'])


PS_FUNC <- function(T, X, dataframe, printR = TRUE){
  form_X <- paste(X, collapse = "+")
  form <- formula(paste(c(T,form_X), collapse = "~"))
  
  logit <- glm(form, family = binomial(link = 'logit'), dataframe) # ,intercept = FALSE
  if(printR){
    print(summary(logit))
  }
  PS <- logit$fitted.values
  
  return(PS)
}

nnm_ATE <- function(dataframe, match_number = 3, match_th = 0.1, use_th = TRUE){
  
  data_temp_1 <- dataframe[dataframe$T == 1, c('Prop_score','Y','Y_0','Y_1')]
  data_temp_0 <- dataframe[dataframe$T == 0, c('Prop_score','Y')]
  cum_effect <- 0
  cum_pehe <- 0
  
  for (i in 1:nrow(data_temp_1)){
    data_temp_0$PSM <- abs(data_temp_0[, 'Prop_score'] - data_temp_1[i, 'Prop_score'])
    
    if(use_th){
      effect_ <- data_temp_1[i,'Y'] - mean(data_temp_0[data_temp_0$PSM <= match_th,'Y'])
    } else{
      effect_ <- data_temp_1[i,'Y'] - mean(data_temp_0[order(data_temp_0$PSM)[1 : match_number],'Y'])
    }
    cum_effect <- cum_effect + effect_
    cum_pehe <- cum_pehe + (effect_ - (data_temp_1[i,'Y_1'] - data_temp_1[i,'Y_0']))^2
  }
  ATE <- cum_effect/nrow(data_temp_1)
  PEHE <- cum_pehe/nrow(data_temp_1)
  return(c(ATE, PEHE))
}
library(gbm)
ITE_Effect <- function(DataFrame, X, PS = 1,weight = 1,model = 'tree'){
  DataFrame_1 <- DataFrame[DataFrame$T == 1,c('T',X,'Y','Y_0','Y_1')]
  DataFrame_0 <- DataFrame[DataFrame$T == 0,c('T',X,'Y','Y_0','Y_1')]
  
  if(length(PS) > 1){
    PS_1 <- DataFrame$T / PS / sum(DataFrame$T / PS)
    PS_0 <- (1 - DataFrame$T) / (1-PS) / sum((1 - DataFrame$T) / (1-PS))
    
    PS_1_ <- PS_1[DataFrame$T == 1]
    PS_0_ <- PS_0[DataFrame$T == 0]
    
    if(model == 'tree'){
      form_X <- paste(X, collapse = "+")
      form <- formula(paste(c('Y',form_X), collapse = "~"))
      
      # t = 1
      model_lm_1 <- gbm(formula = form,data = DataFrame_1, distribution = "gaussian", weights = PS_1_,
                        n.trees = 30, shrinkage = 0.1,             
                        interaction.depth = 5, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_1 <- gbm.perf(model_lm_1, method = "test")
      # t = 0
      model_lm_0 <- gbm(formula = form,data = DataFrame_0, distribution = "gaussian", weights = PS_0_,
                        n.trees = 30, shrinkage = 0.1,             
                        interaction.depth = 5, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_0 <- gbm.perf(model_lm_0, method = "test")
    } else if(model == 'poly'){
      # non-linear
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
    # y1
    if(model == 'tree'){
      predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0,n.trees = best.iter_1)
    } else if(model == 'poly'){
      predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0)
    }
  } else if(length(weight) > 1){
    w_1 <- weight[DataFrame$T == 1]
    w_0 <- weight[DataFrame$T == 0]
    
    if(model == 'tree'){
      form_X <- paste(X, collapse = "+")
      form <- formula(paste(c('Y',form_X), collapse = "~"))
      
      # t = 1
      model_lm_1 <- gbm(formula = form,data = DataFrame_1, distribution = "gaussian", weights = w_1,
                        n.trees = 30, shrinkage = 0.1,             
                        interaction.depth = 5, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_1 <- gbm.perf(model_lm_1, method = "test")
      # t = 0
      model_lm_0 <- gbm(formula = form,data = DataFrame_0, distribution = "gaussian", weights = w_0,
                        n.trees = 30, shrinkage = 0.1,             
                        interaction.depth = 5, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_0 <- gbm.perf(model_lm_0, method = "test")
    } else if(model == 'poly'){
      # non-linear
      
      form <- paste0('Y~',paste0(paste0('poly(', X,sep = ","),'3)',collapse="+"))
      # t = 1
      model_lm_1 <- lm(formula = form,data = DataFrame_1,weights = PS_1_)
      # summary(model_lm_1)
      # t = 0
      model_lm_0 <- lm(formula = form,data = DataFrame_0,weights = PS_0_)
      # summary(model_lm_0)
    }
    
    # pehe
    if(model == 'tree'){
      predict_y_0 <- predict(model_lm_0,newdata = DataFrame_1,n.trees = best.iter_0)
    } else if(model == 'poly'){
      predict_y_0 <- predict(model_lm_0,newdata = DataFrame_1)
    }
    # y1
    if(model == 'tree'){
      predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0,n.trees = best.iter_1)
    } else if(model == 'poly'){
      predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0)
    }
  } else{
    if(model == 'tree'){
      form_X <- paste(X, collapse = "+")
      form <- formula(paste(c('Y',form_X), collapse = "~"))
      
      # t = 1
      model_lm_1 <- gbm(formula = form,data = DataFrame_1, distribution = "gaussian",
                        n.trees = 30, shrinkage = 0.1,             
                        interaction.depth = 5, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_1 <- gbm.perf(model_lm_1, method = "test")
      # t = 0
      model_lm_0 <- gbm(formula = form,data = DataFrame_0, distribution = "gaussian",
                        n.trees = 30, shrinkage = 0.1,             
                        interaction.depth = 5, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_0 <- gbm.perf(model_lm_0, method = "test")
    } else if(model == 'poly'){
      # non-linear
      
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
    # y1
    if(model == 'tree'){
      predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0,n.trees = best.iter_1)
    } else if(model == 'poly'){
      predict_y_1 <- predict(model_lm_1,newdata = DataFrame_0)
    }
  }
  ITE_PRE_1 <- DataFrame_1$Y - predict_y_0
  ITE_true_1 <- DataFrame_1$Y_1 - DataFrame_1$Y_0
  PEHE_1_ <- sum((ITE_PRE_1 - ITE_true_1)^2) 
  
  ITE_PRE_0 <- predict_y_1 - DataFrame_0$Y
  ITE_true_0 <- DataFrame_0$Y_1 - DataFrame_0$Y_0
  PEHE_0_ <- sum((ITE_PRE_0 - ITE_true_0)^2) 
  
  # PEHE
  PEHE = (PEHE_1_ + PEHE_0_) / (length(ITE_PRE_1) + length(ITE_PRE_0))

  return(PEHE)
}

nnm_ITE <- function(dataframe, X, match_number = 5, model = 'tree'){
  
  use_col <- c('T',X,'Prop_score','Y','Y_0','Y_1')
  data_temp_1 <- dataframe[dataframe$T == 1, use_col]
  data_temp_0 <- dataframe[dataframe$T == 0, use_col]
  n1 <- nrow(data_temp_1)
  for (i in 1:n1){
    data_temp_0$PSM <- abs(data_temp_0[, 'Prop_score'] - data_temp_1[i, 'Prop_score'])
    data_temp_1[n1 + i,] <- colMeans(data_temp_0[order(data_temp_0$PSM)[1:match_number],use_col])
  }
  data_temp_1 <- data_temp_1[sample(nrow(data_temp_1)),]
  rownames(data_temp_1) <- 1:nrow(data_temp_1)
  
  PEHE <- ITE_Effect(data_temp_1,X)
  
  return(PEHE)
}

## C
PS_C <- PS_FUNC('T',c('C'),dataframe)
dataframe$Prop_score <- PS_C

## 
R_C <- nnm_ATE(dataframe, match_number = 10, use_th = FALSE)
abs(R_C[1] - ATE_True)

PEHE_C <- nnm_ITE(dataframe, X = c('C'),match_number = 10)
PEHE_C

X1 = c( "I","A","YO","M","TO","Z")
for(x in X1){
  PS <- PS_FUNC('T',c('C',x),dataframe,FALSE)
  dataframe$Prop_score <- PS
  Res <- nnm_ATE(dataframe, match_number = 10, use_th = FALSE)
  ATE <- Res[1];PEHE_F =  Res[2]
  PEHE <- nnm_ITE(dataframe, X = c('C',x),match_number = 10)
  ATE_e <- abs(ATE - ATE_True)
  print(paste(c('################',x,'################'), collapse = ""))
  print(paste(c('ATE_ERROR',ATE_e), collapse = ":"))
  print(paste(c('PEHE',PEHE), collapse = ":"))
}
