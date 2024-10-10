setwd("D:/experiment/Conference Paper/NeurIPS2024/code")
rm(list = ls())
library(MASS)
 
d_index <- 8

dpath            <- file.path("D:/experiment/dataset/regression/") 

Dataset          <- c("elevators_all","bank_all","slice_all","Year_test",
                    "ailerons_all","calhousing","N-twitter","N-TomsHardware")

savepath         <- paste0("D:/experiment/Conference Paper/NeurIPS2024/Result/",
                        paste0("eM-KOFL-",Dataset[d_index],".txt"))

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
J         <- K                       ################ the number of selected hypothesis

eta_g     <- 8
eta_l     <- 1
lambda    <- 0.0001
D         <- 100


for(re in 1:reptimes)
{

  run_time  <- 0  
  u       <- mvrnorm(D,rep(0,feature_tr),diag(feature_tr))   # w--->D*d
  W       <- matrix(0,nrow=K*M, ncol=2*D)         #### store the models of M clients
  zx      <- c(rep(0, 2*D))
  order   <- sample(1:length_tr,length_tr,replace = F)   #dis
  p       <- c(rep(1/K,K))        ################ the sampling probability
  L_loss  <- matrix(0,nrow=M, ncol=K)
  Cum_t   <- c(rep(0, K))
  Cum_loss<- 0
  for(t in 1:(length_tr/10))
  {
    #################### server sample a kernel function ###################################
    It             <- sample(1:K, 1, replace=T,prob=p)
    global_w       <- c(rep(0, 2*D))
    for(j in 1:M)
    {
      global_w     <- global_w + W[K*(j-1)+It,]
    }
    global_w       <- global_w/M
    t1      <- proc.time()                                 #proc.time()
    for(j in 1:M)
    {
      W[K*(j-1)+It,] <- global_w
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
        
        y_i          <- crossprod(W[K*(j-1)+i,],zx)[1,1]
        #################### prediction       
        ins_loss     <- (y_i-ylabel[order[M*(t-1)+j]])^2 + lambda*crossprod(W[K*(j-1)+i,],W[K*(j-1)+i,])
        L_loss[j,i]  <- L_loss[j,i]+ ins_loss
        if( i == It)
          Cum_loss   <- Cum_loss + (y_i-ylabel[order[M*(t-1)+j]])^2
        W[K*(j-1)+i,]    <- (1-2*lambda)*W[K*(j-1)+i,] -eta_l*2*(y_i-ylabel[order[M*(t-1)+j]])*zx
      }
    }
    t2            <- proc.time()
    run_time      <- run_time + (t2 - t1)[3]
    ##################### update the sampling probability using binary search
    for(i in 1:K)
    {
      Cum_t[i] <- exp(-eta_g*sum(L_loss[,i]))
    }
    p <- Cum_t/sum(Cum_t)
  }
  runtime[re]   <- run_time
  errorrate[re] <- Cum_loss/length_tr
}

save_result <- list(
  note     = c(" the next term are:alg_name--dataname--sam_num--run_time--tot_run_time--ave_run_time--err_num--all_err_rate--ave_err_rate--sd_time--sd_err"),
  alg_name = c("eM-KOFL-"),
  dataname = paste0(Dataset[d_index], ".train"),
  sam_num  = length_tr,
  eta_g    = eta_g,
  eta_l    = eta_l,
  lambda   = lambda,
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
sprintf("the average AMR is %f", sum(errorrate)/reptimes)
sprintf("standard deviation of run_time is %.5f in dataset", sd(runtime))
sprintf("standard deviation of AMR is %.5f in dataset", sd(errorrate))
