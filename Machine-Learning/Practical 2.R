#Stochastic Gradient Langevin Dynamics

# call libraries
library(numDeriv)

install.packages("mvtnorm")
library(mvtnorm)
# Load R package for printing
library(knitr)
# Set a seed of the randon number generator
set.seed(2023)

#Supplementary Material
#finding the maximum aposteriori estimator (MAP) of w
#

#Application: Binary classification Provlem
data_generating_model <- function(n,w) {
  d <- 3
  z <- rep( NaN, times=n*d )
  z <- matrix(z, nrow = n, ncol = d)
  z[,1] <- 1.0
  z[,2] <- runif(n, min = -10, max = 10)
  p <- w[1]*z[,1] + w[2]*z[,2] 
  p <- exp(p) / (1+exp(p))
  z[,3] <- rbinom(n, size = 1, prob = p)
  return(z)
}

n_obs <- 10^(6)
w_true <- c(0,1)  
set.seed(2023)
z_obs <- data_generating_model(n = n_obs, w = w_true) 
set.seed(0)
w_true <- as.numeric(glm(z_obs[,3]~ 1 + z_obs[,2],family = "binomial" )$coefficients)
w_true

prediction_rule <- function(x,w) {
  h <- w[1]*x[1]+w[2]*x[2]
  h <- exp(h) / (1.0 + exp(h) )
  return (h)
}

#log PDF of the sampling distribution
log_sampling_pdf <- function(z, w) {
  d <- length(w)
  x <- z[1:d] 
  y <- z[d+1]
  log_pdf <- y * log(prediction_rule(x,w)) +(1-y) * log( 1.0-prediction_rule(x,w) )
  #log_pdf <- dbinom(y, size = 1, prob = prediction_rule(x,w), log = TRUE)
  return( log_pdf )
}

#log PDF of prior distribution
log_prior_pdf <- function(w, mu, Sig2 ) {
  log_pdf <- dmvnorm(w, mean = mu, sigma = Sig2, log = TRUE, checkSymmetry = TRUE)
  return( log_pdf )
}
learning_rate <- function(t, T_0 = 100, T_1 = 500, C_0 = 0.0001, s_0 = 0.5 ) {
  if ( t <= T_0 ) {
    eta <- C_0
  } else if ( (T_0+1 <= t) && (t <= T_1 ) ) {
    eta <- C_0 / ( (t-T_0) ^ s_0 )
  } else {
    eta <- C_0 / ( (T_1-T_0) ^ s_0 )
  }
  return(eta)
}

#1.1 Task
#Coding a Stochastic Gradient Langevin Dynamics (SGLD) algorithm with batch size m = 0.1n,
#and temperature tau = 1.0 that returns the chain of all the {w^(t)} produced.

Tmax <- 500
#
w_seed <- c(-1,0)
#
eta <- 10^(-6)
eta_C <- eta
eta_s <- 0.51
eta_T0 <- 0.3*Tmax
eta_T1 <- 0.6*Tmax
#
batch_size <- 1000
#
tau <- 1.0
#
# Set the seed
w <- w_seed
w_chain <- c(w)
# iterate
t <- 1
Qterm <- 0
#
# iterate
#
while ( (Qterm != 1) ) {
  # counter 
  t <- t+1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # learning rate
  eta <- learning_rate(t, eta_T0, eta_T1, eta_C, eta_s)
  # sub-sample
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  #need the sub-sample to produce the inputs we need for the batch
  # update
  w_new <- w
  ## likelihood
  grad_est_lik <- rep( 0.0, times=length(w) )
  for (j in J) {
    aux_fun <- function(w, z=z_obs[j,]){
      gr <- log_sampling_pdf(z, w)
      return(gr)
    }
    grad_est_lik <- grad_est_lik + numDeriv::grad(aux_fun, w)
  }
  grad_est_lik <- ( n_obs / batch_size) * grad_est_lik
  w_new <- w_new +eta*grad_est_lik ; 
  ## prior
  aux_fun <- function(w){
    d <- length(w)
    gr <- log_prior_pdf(w, rep(0,d), 100*diag(d))
    return(gr)
  }
  w_new <- w_new +eta*numDeriv::grad(aux_fun, w) ;
  ## noise
  w_new <- w_new +sqrt(eta)*sqrt(tau)*rnorm(n = length(w), mean = 0, sd = 1) #the rnorm gives su the epsilon_t
  # record
  w <- w_new
  # termination criterion
  if  ( t >= Tmax ) {
    Qterm <- 1
  }
  # record the produced chain
  w_chain <- rbind(w_chain,w)
}

#1.2 Task 
#Plot the trace plots of chains {w1^t} and {w2^t} against the iteration t
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')

#1.3 Task: 
#Code a SGLD algorithm with more slight changes and now we introduce clipping
Tmax <- 500
#
w_seed <- c(-1,0)
#
eta <- 10^(-2)
eta_C <- eta
eta_s <- 0.51
eta_T0 <- 0.3*Tmax
eta_T1 <- 0.6*Tmax
#
batch_size <- 1000
#
tau <- 1.0
#
# Set the seed
w <- w_seed
w_chain_clipping <- c(w)
# iterate
t <- 1
Qterm <- 0
#
clipping_threshold <- 10
#
# iterate
#
while ( (Qterm != 1) ) {
  # counter 
  t <- t+1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # learning rate
  eta <- learning_rate(t, eta_T0, eta_T1, eta_C, eta_s)
  # sub-sample
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  # update
  w_new <- w
  ## likelihood
  grad_est_lik <- rep( 0.0, times=length(w) )
  for (j in J) {
    aux_fun <- function(w, z=z_obs[j,]){
      gr <- log_sampling_pdf(z, w)
      return(gr)
    }
    grad_est_lik <- grad_est_lik + numDeriv::grad(aux_fun, w)
  }
  grad_est_lik <- ( n_obs / batch_size) * grad_est_lik
  # gradient clipping/rescaling
  norm_grad_est_lik <- sqrt(sum(grad_est_lik^2))
  grad_est_lik <- grad_est_lik * min( 1.0, clipping_threshold/norm_grad_est_lik )
  w_new <- w_new +eta*grad_est_lik ; 
  ## prior
  aux_fun <- function(w){
    d <- length(w)
    gr <- log_prior_pdf(w, rep(0,d), 100*diag(d))
    return(gr)
  }
  w_new <- w_new +eta*numDeriv::grad(aux_fun, w) ;
  ## noise
  w_new <- w_new +sqrt(eta)*sqrt(tau)*rnorm(n = length(w), mean = 0, sd = 1)
  # record 
  w <- w_new
  # termination criterion
  if  ( t >= Tmax ) {
    Qterm <- 1
  }
  # record the produced chain
  w_chain_clipping <- rbind(w_chain_clipping,w)
}
plot(w_chain_clipping[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain_clipping[,2], type='l') +
  abline(h=w_true[2], col='red')

#1.4 Task 
#back to the code without using the clipping gradient
#copy the tail end of the generated chain w(t)
#after discarding the burn in, as ‘w_chain_output’.
T_burnin <- as.integer(0.6*Tmax)
w_chain_output <- w_chain[T_burnin:Tmax,]

#1.5 Task
#Plot the histograms plots of output chains {w1(t)}
#and {w2^(t)} for the estimation of the marginal 
#posterior distributions of the dimensions of w.
hist(w_chain_output[,1]) 
hist(w_chain_output[,2]) 

#1.6 Task 
#computing an estimator of the w_chain outputs
w_est = mean(rowSums(w_chain_output))
w_est

#1.7 Task
#Use the prediction rule: hw(x)=exp(x⊤w)1+exp(x⊤w)
#and the estimates for w in order to classify the example with feature xnew=c(1,0.5)
#particularly, computing point estimates
x_new <- c(1,0.5)
T <- dim(w_chain_output)[1]
h_est <- 0.0 
for (t in 1:T) {
  h_est <- h_est + prediction_rule( x_new , w_chain_output[t,] )
}
h_est <- h_est / T
h_est

#now compute the point estimate by plugging in the estimate
x_new <- c(1,0.5)
w_est <- colMeans(w_chain_output)
h_est <- prediction_rule( x_new , w_est )
h_est

#Now, estimate the pdf of the prediction rule at the example 
#with feature xnew=c(1,0.5) (to represent the uncertainty) by using a histogram
x_new <- c(1,0.5)
T <- dim(w_chain_output)[1]
h_chain <- c()
for (t in 1:T) {
  h_chain <- c( h_chain , 
                prediction_rule( x_new , w_chain_output[t,] ) 
  )
}
hist(h_chain)

#Additional Tasks 
#1 Mixture Model  
data_generating_model <- function(n,w) {
  z <- rep( NaN, times=n )
  p1 <- 0.5 
  p2 <- 1.0-p1
  w <- 5
  phi <-  20
  sig2 <- 5
  lab <- as.numeric(runif(n_obs)>p1)
  z <- lab*rnorm(n, mean = w, sd = sqrt(sig2)) + (1-lab)*rnorm(n, mean = phi-w, sd = sqrt(sig2))
  return(z)
}
n_obs <- 10^(6)
w_true <- 5 
set.seed(2023)
z_obs <- data_generating_model(n = n_obs, w = w_true) 
set.seed(0)
hist(z_obs)

#1.1 Task
#pdf of the sampling distribution
log_sampling_pdf <- function(z, w, p1 = 0.5, phi=20, sig2 = 5) {
  log_sampling_pdf <- p1*dnorm(z, mean = w, sd = sqrt(sig2), log = FALSE)
  log_sampling_pdf <- log_sampling_pdf + (1-p1)*dnorm(z, mean = phi-w, sd = sqrt(sig2), log = FALSE)
  log_sampling_pdf <- log(log_sampling_pdf) ;
  return(log_sampling_pdf)
}

#1.2 Task 
#pdf of prior distribution (Normal with mean = 0 and variance = 100)
log_prior_pdf <- function(w, mu = 0.0, sig2 = 100 ) {
  log_pdf <- dnorm(w, mean = mu, sd = sqrt(sig2), log = TRUE)
  return( log_pdf )
}

#1.3 Task
#learning rate function again!
learning_rate <- function(t, T_0  , T_1  , C_0  , s_0   ) {
  if ( t <= T_0 ) {
    eta <- C_0
  } else if ( (T_0+1 <= t) && (t <= T_1 ) ) {
    eta <- C_0 / ( (t-T_0) ^ s_0 )
  } else {
    eta <- C_0 / ( (T_1-T_0) ^ s_0 )
  }
  return(eta)
}

#1.4 Task
#Compute the gradient of the log pdf of the sampling distribution 
#with respect to w at point w=4.0 (at the 1st example; i.e. z1).
#Do this by using the function ‘grad{numDeriv}’ from the R package numDeriv.
w <- 4.0
aux_fun <- function(w, z = z_obs[1]) {
  return( log_sampling_pdf(z, w) ) 
}
gr <- numDeriv::grad( aux_fun, w )
gr

#1.5 Task 
#SGLD algorithm with new inputs
Tmax <- 500
#
w_seed <- 0.0
#
eta <- 10^(-2)
eta_C <- eta
eta_s <- 0.51
eta_T0 <- 0.3*Tmax
eta_T1 <- 0.6*Tmax
#
batch_size <- 1000
#
tau <- 1.0
#
# Set the seed
w <- w_seed
w_chain_clipping <- c(w)
# iterate
t <- 1
Qterm <- 0
#
clipping_threshold <- 10
#
# iterate
#
while ( (Qterm != 1) ) {
  # counter 
  t <- t+1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # learning rate
  eta <- learning_rate(t, eta_T0, eta_T1, eta_C, eta_s)
  # sub-sample
  J <- sample.int( n = n_obs, size = batch_size, replace = FALSE)
  # update
  w_new <- w
  ## likelihood
  grad_est_lik <- rep( 0.0, times=length(w) )
  for (j in J) {
    aux_fun <- function(w, z=z_obs[j]){
      gr <- log_sampling_pdf(z, w)
      return(gr)
    }
    grad_est_lik <- grad_est_lik + numDeriv::grad(aux_fun, w)
  }
  grad_est_lik <- ( n_obs / batch_size) * grad_est_lik
  # gradient clipping/rescaring
  norm_grad_est_lik <- sqrt(sum(grad_est_lik^2))
  grad_est_lik <- grad_est_lik * min( 1.0, clipping_threshold/norm_grad_est_lik )
  w_new <- w_new +eta*grad_est_lik ; 
  ## prior
  aux_fun <- function(w){
    d <- length(w)
    gr <- log_prior_pdf(w, rep(0,d), 100*diag(d))
    return(gr)
  }
  w_new <- w_new +eta*numDeriv::grad(aux_fun, w) ;
  ## noise
  w_new <- w_new +sqrt(eta)*sqrt(tau)*rnorm(n = length(w), mean = 0, sd = 1)
  # record
  w <- w_new
  # termination criterion
  if  ( t >= Tmax ) {
    Qterm <- 1
  }
  # record the produced chain
  w_chain_clipping <- rbind(w_chain_clipping,w)
}
plot(w_chain_clipping, type = 'l')
hist(w_chain_clipping)
