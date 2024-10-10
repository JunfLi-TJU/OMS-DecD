setwd("D:/experiment/Conference Paper/NeurIPS2024/code")
rm(list = ls())
library(MASS)
 
d_index <- 8

dpath            <- file.path("D:/experiment/dataset/regression/") 

Dataset          <- c("elevators_all","bank_all","slice_all","Year_test",
                    "ailerons_all","calhousing","N-twitter","N-TomsHardware")

savepath         <- paste0("D:/experiment/Conference Paper/NeurIPS2024/Result/",
                        paste0("FOMD-OMS-Lin-K-",Dataset[d_index],".txt"))

traindatapath    <- file.path(dpath, paste0(Dataset[d_index], ".train"))                
traindatamatrix  <- as.matrix(read.table(traindatapath))
trdata           <- traindatamatrix[ ,-1]
ylabel           <- traindatamatrix[ ,1]

length_tr        <- nrow(trdata) 
length_tr        <- floor(length_tr/10)*10
feature_tr       <- ncol(trdata)  

reptimes  <- 10

runtime   <- c(rep(0, reptimes))
errorrate <- c(rep(0, reptimes))


U         <- seq(0.1,1.0,by=0.1)
M         <- 10                   ################ the number of clients
K         <- length(U)            ################ the number of candidate hypothesis spaces
J         <- K                    ################ the number of selected hypothesis

Y         <- max(ylabel)
eta1      <- sqrt(log(K*length_tr))/sqrt(4*length_tr*(1+(K-J)/(J*M-M)))
eta_t     <- 1*min(eta1,(J-1)/(2*K-2*J))
C         <- (U+Y)^2
alpha     <- 1

Yt        <- matrix(0,nrow=M, ncol=J)
W         <- matrix(0,nrow=K, ncol=feature_tr)           #### store the models of M clients
grad      <- matrix(0,nrow=M*J, ncol=feature_tr)         #### store the gradient of selected models
grad_i    <- c(rep(0,feature_tr)) 
  
for(re in 1:reptimes)
{
  W       <- matrix(0,nrow=K, ncol=feature_tr)           #### store the models of M clients
  order   <- sample(1:length_tr,length_tr,replace = F)   #dis
  Cum_loss<- 0
  flag    <- 0
  p       <- c(rep(alpha/(sqrt(K*length_tr)),K))   ################ the sampling probability
  p[1]    <- 1-alpha*sqrt(K)/sqrt(length_tr)+alpha/(sqrt(K*length_tr))
  run_time  <- 0

  
  for(t in 1:(length_tr/10))
  {
    ct         <- c(rep(0,K))
    ins_loss   <- matrix(0,nrow=M, ncol=K)
    t1         <- proc.time()                                 #proc.time()
    for(j in 1:M)
    {
      #################### sampling hypotheses ###################################
      It       <- sample(1:K, 1, replace=T,prob=p)
      
      #################### read the models of the j-th client and the data received by the j-th client 
      xt       <- trdata[order[M*(t-1)+j],]
      #################### prediction 
      for(i in 1:K)
      {
        Yt[j,i]  <- crossprod(W[i,],xt)[1,1]
        #################### prediction       
        ins_loss[j,i]   <- (Yt[j,i]-ylabel[order[M*(t-1)+j]])^2
        if(i == It)
          Cum_loss <- Cum_loss + ins_loss[j,i]
        grad[J*(j-1)+i,] <- 2*(Yt[j,i]-ylabel[order[M*(t-1)+j]])*xt
      }
    }
    t2            <- proc.time()
    run_time      <- run_time + (t2 - t1)[3]
    ################## global updating
    for(i in 1:K)
    {
      G_i        <- 2*(U[i]+Y)
      lambda_ti  <- U[i]/(2*G_i*sqrt(1+(K-J)/((J-1)*M)))/sqrt(max((K-J)^2/(J-1)^2,t))
      grad_i     <- c(rep(0,feature_tr))
      for(j in 1:M)
      {
        ct[i]    <- ct[i] + ins_loss[j,i]
        grad_i   <- grad_i + grad[J*(j-1)+i,]
      }
      ct[i] <- ct[i]/M
      W[i,] <- W[i,] -lambda_ti*grad_i/M
      norm_i <- sqrt(crossprod(W[i,],W[i,])[1,1])
      if(norm_i>U[i])
      {
        W[i,] <- W[i,]*U[i]/norm_i
      }
    }
    
    ##################### update the sampling probability using binary search
    upper_lambda   <- 0    #### lambda^\ast < 0
    lambda_ast     <- upper_lambda
    tem            <- p*exp(-eta_t*(lambda_ast+ct)/C)
    sum_barpt      <- sum(tem)
    if(abs(sum_barpt - 1)>1e-5)
    {
      lower_lambda <- -max(ct)
      lambda_ast   <- lower_lambda
      tem          <- p*exp(-eta_t*(lambda_ast+ct)/C)
      sum_barpt    <- sum(tem)
      if(abs(sum_barpt - 1)>1e-5)
      {
        lambda_ast <- (lower_lambda+upper_lambda)/2 
        tem        <- p*exp(-eta_t*(lambda_ast+ct)/C)
        sum_barpt  <- sum(tem)
      }
      while(abs(sum_barpt - 1)>1e-5)
      {
        flag <- flag +1
        if(sum_barpt<1)      ############ lambda^ast should decrease
        {
          upper_lambda  <- lambda_ast
        }else{                        ############ lambda^ast should increase
          lower_lambda  <- lambda_ast
        }
        lambda_ast      <- (lower_lambda+upper_lambda)/2
        tem             <- p*exp(-eta_t*(lambda_ast+ct)/C)
        sum_barpt       <- sum(tem)
      }
    }
    p <- tem
  }
  runtime[re]   <- run_time
  errorrate[re] <- Cum_loss/length_tr
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--sam_num--run_time--tot_run_time--ave_run_time--err_num--all_err_rate--ave_err_rate--sd_time--sd_err"),
  alg_name = c("FOMD-OMS-Lin-K"),
  dataname = paste0(Dataset[d_index], ".train"),
  sam_num  = length_tr,
  run_time = as.character(runtime),
  tot_run_time = sum(runtime),
  ave_run_time = sum(runtime)/reptimes,
  err_num      = errorrate,
  ave_err_rate = sum(errorrate)/reptimes,
  sd_time      <- sd(runtime),
  sd_err       <-sd(errorrate)
)
write.table(save_result,file=savepath,row.names =TRUE, col.names =FALSE, quote = T)

sprintf("the number of sample is %d", length_tr)
sprintf("total running time is %.1f in dataset", sum(runtime))
sprintf("average running time is %.1f in dataset", sum(runtime)/reptimes)
sprintf("the average MSE is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of run_time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of MSE is %.5f in dataset", sd(errorrate))
