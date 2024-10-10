setwd("D:/experiment/Conference Paper/NeurIPS2024/code")
rm(list = ls())
library(MASS)
 
d_index <- 8

dpath            <- file.path("D:/experiment/dataset/regression/") 

Dataset          <- c("elevators_all","bank_all","slice_all","Year_test",
                    "ailerons_all","calhousing","N-twitter","N-TomsHardware")

savepath         <- paste0("D:/experiment/Conference Paper/NeurIPS2024/Result/",
                        paste0("POF-MKL-",Dataset[d_index],".txt"))

traindatapath    <- file.path(dpath, paste0(Dataset[d_index], ".train"))                
traindatamatrix  <- as.matrix(read.table(traindatapath))
trdata           <- traindatamatrix[ ,-1]
ylabel           <- traindatamatrix[ ,1]

length_tr        <- nrow(trdata) 
length_tr        <- floor(length_tr/10)*10
feature_tr       <- ncol(trdata)  

x         <-seq(-1,6,1)
sigma     <- 2^(x)
len_sigma <- length(sigma)

reptimes  <- 10

runtime   <- c(rep(0, reptimes))
errorrate <- c(rep(0, reptimes))
M         <- 10                      ################ the number of clients
K         <- len_sigma               ################ the number of candidate hypothesis spaces
J         <- 2                       ################ the number of selected hypothesis
m         <- K/J

eta_i     <- 0.5
eta       <- 0.5
lambda    <- 0.0001
D         <- 100
q         <- c(rep(1/m,m))

for(re in 1:reptimes)
{

  run_time  <- 0
  u       <- mvrnorm(D,rep(0,feature_tr),diag(feature_tr))   # w--->D*d
  W       <- matrix(0,nrow=K, ncol=2*D)         #### store the models of M clients
  zx      <- c(rep(0, 2*D))
  order   <- sample(1:length_tr,length_tr,replace = F)   #dis
  p       <- matrix(1,nrow=M, ncol=K)           ################ the weight
  Cum_loss<- 0
  for(t in 1:(length_tr/10))
  {
    #################### server sample a kernel function ###################################
    grad    <- matrix(0,nrow=M*K, ncol=2*D)         #### store the gradient of selected models
    t1      <- proc.time()                                 #proc.time()
#   t1 <- Sys.time() 
    for(j in 1:M)
    {
      haty_i   <- 0
      yt       <- c(rep(0, K))
      coe      <- p[j,]/sum(p[j,])
      st       <- sample(1:m, 1, replace=T,prob=q)
      
      tem_p    <- rev(sort(p[j,]))
      if(tem_p[2*st] != tem_p[2*st-1])
      {
        It       <- which(p[j,]==tem_p[2*st])[1] 
        Jt       <- which(p[j,]==tem_p[2*st-1])[1]
      }else
      {
        It       <- which(p[j,]==tem_p[2*st])[1] 
        Jt       <- which(p[j,]==tem_p[2*st-1])[2]
      }        
      
      #################### read the models of the j-th client and the data received by the j-th client 
      xt             <- trdata[order[M*(t-1)+j],]
      tem1           <- u%*%xt
      #################### prediction 
      for(i in 1:K)
      {
        ################## compute random features
        tem          <- tem1/sigma[i]
        coszx        <- cos(tem)/sqrt(D)
        sinzx        <- sin(tem)/sqrt(D)
        zx           <- c(coszx,sinzx)
        #################### prediction          
        yt[i]        <- crossprod(W[i,],zx)[1,1]
        haty_i       <- haty_i + yt[i]*coe[i]
      }
      Cum_loss   <- Cum_loss + (haty_i-ylabel[order[M*(t-1)+j]])^2 
      for(i in 1:K)
      {
        ins_loss     <- (yt[i]-ylabel[order[M*(t-1)+j]])^2 + lambda*crossprod(W[i,],W[i,])
        p[j,i]       <- p[j,i]*exp(-eta_i*ins_loss)        
        if( (i == It) ||(i == Jt))
        {
          grad[K*(j-1)+i,]   <- (2*lambda*W[i,] +2*(yt[i]-ylabel[order[M*(t-1)+j]])*zx)*m
        }
      }
    }
    t2            <- proc.time()
#    t2 <- Sys.time() 
    run_time      <- run_time + (t2 - t1)[3]
#    run_time      <- run_time + (t2 - t1)
    ##################### update the sampling probability using binary search
    for(i in 1:K)
    {
      grad_i     <- c(rep(0,2*D))
      for(j in 1:M)
      {
        grad_i   <- grad_i + grad[K*(j-1)+i,]
      }
      W[i,] <- W[i,] -eta*grad_i/M
    }
  }

  runtime[re]   <- run_time
  errorrate[re] <- Cum_loss/length_tr
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--sam_num--run_time--tot_run_time--ave_run_time--err_num--all_err_rate--ave_err_rate--sd_time--sd_err"),
  alg_name = c("POF-MKL-"),
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
