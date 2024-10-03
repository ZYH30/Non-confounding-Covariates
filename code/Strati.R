dataframe <- read.table('../data/data-discrete.csv', header = TRUE, sep = ',')
colnames(dataframe)
ATE_True <- mean(dataframe$Y_1 - dataframe$Y_0)
ATE_Dire <- mean(dataframe[dataframe$T == 1,'Y']) - mean(dataframe[dataframe$T == 0,'Y'])
Cor_init <- cor(dataframe[,'T'],dataframe[,'C'],method= 'pearson')
ks.test(dataframe[dataframe$T == 1,'C'],dataframe[dataframe$T == 0,'C'])

PS_FUNC <- function(T, X, dataframe, printR = TRUE){
  # form_X <- paste(c(paste(X, collapse = "+"),0), collapse = "+")
  form_X <- paste(X, collapse = "+")
  form <- formula(paste(c(T,form_X), collapse = "~"))
  # print(form)
  logit <- glm(form, family = binomial(link = 'logit'), dataframe) # ,intercept = FALSE
  if(printR){
    print(summary(logit))
  }
  PS <- logit$fitted.values
  
  return(PS)
}


Strat_Effect <- function(dataframe,segment,X,isITE = TRUE){
  # segment 
  max_pps = max(dataframe$Prop_score)
  min_pps = min(dataframe$Prop_score)
  segLength <- round((max_pps -min_pps) / segment, 2)
  ATE_ST = 0
  PEHE_ST = 0
  COR_ST = 0
  for(i in 1:segment){
    ind_ <- (dataframe$Prop_score >= min_pps + (i - 1) * segLength) & (dataframe$Prop_score <= min_pps + (i) * segLength)
    GrounpData <- dataframe[ind_,]
    Effect_ <- mean(GrounpData[GrounpData$T == 1, 'Y']) - mean(GrounpData[GrounpData$T == 0, 'Y'])
    if(isITE){
      PEHE_ <- ITE_Effect(GrounpData,X)
      PEHE_ST <- PEHE_ST + PEHE_ * nrow(GrounpData) / nrow(dataframe)
    }
    Cor_ <- abs(cor(GrounpData[,'T'],GrounpData[,'C'],method= 'pearson'))
    ks_ <- ks.test(dataframe[GrounpData$T == 1,'C'],dataframe[GrounpData$T == 0,'C'])
    
    COR_ST <- COR_ST + Cor_ * nrow(GrounpData) / nrow(dataframe)
    ATE_ST <- ATE_ST + Effect_ * nrow(GrounpData) / nrow(dataframe)
  }
  
  if(is.na(ATE_ST)){
    print('Segment is too big!')
  } else{
    return(c(ATE_ST,COR_ST,PEHE_ST))
  }
}
library(gbm)
ITE_Effect <- function(DataFrame, X, PS = 1,weight = 1,model = 'tree'){
  # split
  DataFrame_1 <- DataFrame[DataFrame$T == 1,c('T',X,'Y','Y_0','Y_1')]
  DataFrame_0 <- DataFrame[DataFrame$T == 0,c('T',X,'Y','Y_0','Y_1')]
  
  if(length(PS) > 1){
    
    PS_1 <- DataFrame$T / PS / sum(DataFrame$T / PS)
    PS_0 <- (1 - DataFrame$T) / (1-PS) / sum((1 - DataFrame$T) / (1-PS))
    
    # weight
    PS_1_ <- PS_1[DataFrame$T == 1]
    PS_0_ <- PS_0[DataFrame$T == 0]
    
    if(model == 'tree'){
      form_X <- paste(X, collapse = "+")
      form <- formula(paste(c('Y',form_X), collapse = "~"))
      
      # t = 1
      model_lm_1 <- gbm(formula = form,data = DataFrame_1, distribution = "gaussian", weights = PS_1_,
                        n.trees = 100, shrinkage = 0.1,             
                        interaction.depth = 3, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_1 <- gbm.perf(model_lm_1, method = "test")
      
      # t = 0
      model_lm_0 <- gbm(formula = form,data = DataFrame_0, distribution = "gaussian", weights = PS_0_,
                        n.trees = 100, shrinkage = 0.1,             
                        interaction.depth = 3, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_0 <- gbm.perf(model_lm_0, method = "test")
      
    } else if(model == 'poly'){
      # nonlinear
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
    # weight
    w_1 <- weight[DataFrame$T == 1]
    w_0 <- weight[DataFrame$T == 0]
    
    if(model == 'tree'){
      form_X <- paste(X, collapse = "+")
      form <- formula(paste(c('Y',form_X), collapse = "~"))
      
      # t = 1
      model_lm_1 <- gbm(formula = form,data = DataFrame_1, distribution = "gaussian", weights = w_1,
                        n.trees = 100, shrinkage = 0.1,             
                        interaction.depth = 3, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_1 <- gbm.perf(model_lm_1, method = "test")
      
      # t = 0
      model_lm_0 <- gbm(formula = form,data = DataFrame_0, distribution = "gaussian", weights = w_0,
                        n.trees = 100, shrinkage = 0.1,             
                        interaction.depth = 3, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
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
                        n.trees = 100, shrinkage = 0.1,             
                        interaction.depth = 3, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_1 <- gbm.perf(model_lm_1, method = "test")
      
      # t = 0
      model_lm_0 <- gbm(formula = form,data = DataFrame_0, distribution = "gaussian",
                        n.trees = 100, shrinkage = 0.1,             
                        interaction.depth = 3, bag.fraction = 0.8, train.fraction = 0.6,  
                        n.minobsinnode = 5, cv.folds = 1, keep.data = TRUE, 
                        verbose = FALSE, n.cores = 1)
      best.iter_0 <- gbm.perf(model_lm_0, method = "test")
    } else if(model == 'poly'){
      # nonlinear
      
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

## C
PS_C <- PS_FUNC('T',c('C'),dataframe)
dataframe$Prop_score <- PS_C
# segment
R_C <- Strat_Effect(dataframe,segment = 10,X = c('C'))
ATE_C = R_C[1];Cor_C =  R_C[2]
abs(ATE_C - ATE_True)
# Cor_C
R_C[3]

X1 = c( "I","A","YO")
for(x in X1){
  PS <- PS_FUNC('T',c('C',x),dataframe,FALSE)
  dataframe$Prop_score <- PS
  # segment
  Res <- Strat_Effect(dataframe,segment = 10,X = c('C',x))
  ATE <- Res[1];Cor =  Res[2];PEHE = Res[3]
  ATE_e <- abs(ATE - ATE_True)
  print(paste(c('################',x,'################'), collapse = ""))
  print(paste(c('ATE_ERROR',ATE_e), collapse = ":"))
  print(paste(c('COR_PROCESS',Cor), collapse = ":"))
  print(paste(c('PEHE',PEHE), collapse = ":"))
}

X2 = c("M","TO","Z")
for(x in X2){
  PS <- PS_FUNC('T',c('C',x),dataframe,FALSE)
  dataframe$Prop_score <- PS
  # segment
  Res <- Strat_Effect(dataframe,segment = 2,X = c('C',x),isITE = TRUE)
  PEHE = Res[3]
  
  Res <- Strat_Effect(dataframe,segment = 10,X = c('C',x),isITE = FALSE)
  ATE <- Res[1]
  ATE_e <- abs(ATE - ATE_True)
  print(paste(c('################',x,'################'), collapse = ""))
  print(paste(c('ATE_ERROR',ATE_e), collapse = ":"))
  print(paste(c('COR_PROCESS',Cor), collapse = ":"))
  print(paste(c('PEHE',PEHE), collapse = ":"))
}