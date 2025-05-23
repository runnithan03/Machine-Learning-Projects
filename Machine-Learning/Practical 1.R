#Supplementary material for computer practical class 1
#function for the gradient descent algorithm
#eta <- 0.5
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
t0 = 10
n_obs = 10
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update
  eta <-  t0/t
  erf_fun <- function(w, z = z_obs, n=n_obs) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  w_chain <- rbind(w_chain, w)
  # step 2: check for termination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
mean(w) #1.167916

#look at HTML file: Computer Practical Sheet 1: Stochastic learning methods
#for introduction and the goal (roughly speaking...) of approximating w* and w**

#Stochastic Learning methods on a binary classification problem
rm(list=ls())

# call libraries
install.packages("numDeriv")
library(numDeriv)

install.packages("nloptr")
library(nloptr)

# Load R package for printing
library(knitr)

# Set a seed of the randon number generator
set.seed(2023)

#Application: Binary classification problem
#he introduces the prediction rule on a sampling distribution

#The dataset Sn {zi = (xi, yi)} is generatde from the data generation probability
#g(.) provided below as a routine, we pretend that we do not know g(.).
data_generating_model <- function(n,w) {
  z <- rep( NaN, times=n*2 )
  z <- matrix(z, nrow = n, ncol = 2)
  z[,1] <- runif(n, min = -10, max = 10)
  p <- w[1] + w[2]*z[,1] 
  p <- exp(p) / (1+exp(p))
  z[,2] <- rbinom(n, size = 1, prob = p)
  return(z)
}

#Let the dataset Sn have size n = 500
#assume that the real value for the unknown parameters w is w_true = (0.0,1.0)^T
#The dataset containing the examples to train the model are generated below, and stores in the array zobs.
set.seed(2023)
n_obs <- 500
w_true <- c(0,1)  
z_obs <- data_generating_model(n = n_obs, w = w_true) 
w_true <- as.numeric(glm(z_obs[,2]~ 1+ z_obs[,1],family = "binomial" )$coefficients)

#then we utilise the prediction rule
#The function prediction_rule(x,w) that returns the rule h
#where x is the input argument and w is the unknown parameter is given below.
prediction_rule <- function(x,w) {
  h <- w[1]+w[2]*x
  h <- exp(h) / (1.0 + exp(h) ) #prediction rule formula
  return (h)
}

#We then consider a likelihood function and consider a loss function
#The code for the loss function is provided below as loss_fun(w,z) that computes the loss function, where z=(x,y)
#is one example (observation) and w is the unknown parameter.
loss_fun <- function(w,z) {
  x = z[1]
  y = z[2]
  h <- prediction_rule(x,w)
  ell <- -y*log(h) - (1-y)*log(1-h) #loss function
  return (ell)
}

#We then define the risk function under the data generation model g
#And the empirical function using the loss function
empirical_risk_fun <- function(w,z,n) {
  x = z[,1]
  y = z[,2]
  R <- 0.0
  for (i in 1:n) {
    R <- R + loss_fun(w,z[i,])
  }
  R <- R / n
  return (R)
}

#computes learning rate sequence
learning_rate <- function(t,t0=3){
  eta = t0/t
  return(eta)
}

#returns the gradient of the loss function at parameter value w, and at e.g. z = (x,y)
grad_loss_fun <- function(w,z){
  x = z[1]
  y = z[2]
  h <- prediction_rule(x,w)
  grd <- c(h-y, (h-y)*x)
  return (grd)  
}
#just remember the above formula I guess

#returns the gradient of the risk function at parameter value w, 
#and using the data set z of size nx2; z = (x,y)
grad_risk_fun <- function(w,z,n) {
  grd <- 0.0
  for (i in 1:n) {
    grd <- grd + grad_loss_fun(w,z[i,]) #takes only x values
  }
  grd <- grd / n
  return (grd)
}

#0.1 Task without numDeriv::grad(fun,w)
w_grad_risk <- c(-0.1,1.5)
grad_risk_fun(w_grad_risk,z_obs,n_obs)
#[1] -0.005427957  0.045414831

#with numDeriv::grad(fun,w)
erf_fun <- function(w, z = z_obs, n=n_obs) {
  return( empirical_risk_fun(w, z, n) ) 
} 
numDeriv::grad(erf_fun,w_grad_risk)
#[1] -0.005427957  0.045414831

#0.2 Task - I don't think he put this in the solutions
?grad
w_empirical_risk <- c(-0.3,3)
erf_fun <- function(w, z = z_obs, n=n_obs) {
  return( empirical_risk_fun(w, z, n) ) 
} 
#we set z = z_obs and n = n_obs so erf_fun can consider them as default values
#this means we can have 1 argument which is w, so numDeriv::grad works!
numDeriv::grad(erf_fun,w_empirical_risk)
#[1] -0.01044497  0.07158301

#or we can do for 0.2 (without numDeriv::grad function):
w_empirical_risk <- c(-0.3,3)
grad_risk_fun(w_empirical_risk, z_obs,n_obs) #they are roughly the same

#1 Gradient descent
#1.1 Task
eta <- 0.5
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  erf_fun <- function(w, z = z_obs, n=n_obs) {
    return( empirical_risk_fun(w, z, n) ) 
  }#changes the number of input arguments required from 3 to 1
  w <- w - eta * numDeriv::grad( erf_fun, w ) #iteration formula with gradient of Rs(w)
  w_chain <- rbind(w_chain, w) #gives us w1(t) and w2(t)
  # step 2: check for termination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
#how does this work?
#another example of the above:
# eta <- 0.1
# Tmax <- 1000
# w_seed <- c(-0.1,1.5)
# w <- w_seed
# w_chain <- c()
# Qstop <- 0
# t <- 0
# while ( Qstop == 0 ) {
# # # counter
#    t <- t +  1
#    cat( t ) ; cat( ' ' ) ## counter added for display reasons
#    eta <- learning_rate( t )
#    # step 1: update
#    w <- w - eta * grad_risk_fun( w, z_obs, n_obs )
#    w_chain <- rbind(w_chain, w)
#    # step 2: check for termination terminate
#    if ( t>= Tmax ) {
#      Qstop <- 1
#    }
#  }
# mean(w)

#1.2 Task
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red') 
#I was basically right but this just gives you a smooth plot
#same idea for this
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')

#1.3 Task 
eta = c(0.001,1)
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  erf_fun <- function(w, z = z_obs, n=n_obs) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  w_chain <- rbind(w_chain, w)
  # step 2: check for termination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
par(mfrow = c(1,2))
plot(w_chain[,1], type='l')
abline(h=w_true[1], col='red') #I was basically right but this just gives you a smooth plot
#same idea for this
plot(w_chain[,2], type='l')
abline(h=w_true[2], col='red')
#does number of iterations need to be changed?

#1.4 Task
learning_rate <- function(t,t0) {
  return(t0/t)
}
t0<- 10
Tmax <- 300
w_seed <- c(-0.3,3.0)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  eta <- learning_rate( t, t0 )
  erf_fun <- function(w, z = z_obs, n=n_obs) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, z_obs, n_obs )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red') 
#why doesn't abline plot here
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')

#2 Batch Stochastic Gradient Descent
set.seed(2023)
n_obs <- 1000000
w_true <- c(0,1)  
z_obs <- data_generating_model(n = n_obs, w = w_true) #This is our sample S, so what we take our subsamples from
w_true <- as.numeric(glm(z_obs[,2]~ 1+ z_obs[,1],family = "binomial" )$coefficients)

#2.1 Task 

?sample.int
m <- 10
eta <- 0.5
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  }
  else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')
#what has changed in the function for the stochastic gradient descent?

#2.2 Task
m <- 80
eta <- 0.5
Tmax <- 300
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  }
  else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')
#ANSWER; As discussed in the lectures, the bigger the batch size the 
#smaller the variation of the gradient, hence the error is smaller, and convergence is quicker

#2.3.1 Ada Grad
#2.3 Additional Tasks
#What would you do if you wish the learning rate to be automatically adjusted?
#(i)
m <- 1
eta <- 1.0
Tmax <- 500
w_seed <- c(-0.3,3.0)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
G <- rep(0.0,times=length(w))
eps <- 10^(-6)
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  } else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  g <- numDeriv::grad( erf_fun, w )
  G <- G + g^2
  w <- w - eta * (1.0/sqrt(G+eps)) * g
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')

#(ii)
m <- 100
eta <- 1.0
Tmax <- 500
w_seed <- c(-0.3,3)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
G <- rep(0.0,times=length(w))
eps <- 10^(-6)
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  } else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  g <- numDeriv::grad( erf_fun, w )
  G <- G + g^2
  w <- w - eta * (1.0/sqrt(G+eps)) * g
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')

#2.3.2 Projection
#(i)
#What would you do if the parametric space / hypothesis class is constrained?
install.packages("nloptr")
library(nloptr)
# out <- nloptr(x0=..,
#               eval_f=..., #
#               eval_grad_f=...,
#               eval_g_ineq = ...,
#               eval_jac_g_ineq = ...,
#               w_now=...,
#               opts = list("algorithm" = "NLOPT_LD_MMA",
#                           "xtol_rel"=1.0e-8)
# out$solution

boundary <- 2.0 # this is the value |w|_{2}^{2} <= boundary
# auxiliary functions to compute the projection
eval_f0 <- function( w_proj, w_now ){ 
  return( sqrt(sum((w_proj-w_now)^2)) )
}
eval_grad_f0 <- function( w, w_now ){ 
  return( c( 2*(w[1]-w_now[1]), 2*(w[2]-w_now[2]) ) )
}
eval_g0 <- function( w_proj, w_now) {
  return( sum(w_proj^2) -(boundary)^2 )
}
eval_jac_g0 <- function( w, w_now ) {
  return(   c(2*w[1],2*w[2] )  )
}

w <- c(-0.1,0.3)
out <- nloptr(x0=c(0.0,0.0),
              eval_f=eval_f0,
              eval_grad_f=eval_grad_f0,
              eval_g_ineq = eval_g0,
              eval_jac_g_ineq = eval_jac_g0, 
              w_now=w,
              opts = list("algorithm" = "NLOPT_LD_MMA",
                          "xtol_rel"=1.0e-8) 
)
out$solution

#(ii)
boundary <- 2.0 # this is the value |w|_{2}^{2} <= boundary
# auxiliary functions to compute the projection
eval_f0 <- function( w_proj, w_now ){ 
  return( sqrt(sum((w_proj-w_now)^2)) )
}
eval_grad_f0 <- function( w, w_now ){ 
  return( c( 2*(w[1]-w_now[1]), 2*(w[2]-w_now[2]) ) )
}
eval_g0 <- function( w_proj, w_now) {
  return( sum(w_proj^2) -(boundary)^2 )
}
eval_jac_g0 <- function( w, w_now ) {
  return(   c(2*w[1],2*w[2] )  )
}
m <- 1
eta <- 0.1
Tmax <- 1000
w_seed <- c(-0.3,3.0)

w <- w_seed
w_chain <- c(w_seed)
Qstop <- 0 
t <- 0
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  } else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * numDeriv::grad( erf_fun, w )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  # step 1.5 projection
  out <- nloptr(x0=c(0.0,0.0),
                eval_f=eval_f0,
                eval_grad_f=eval_grad_f0,
                eval_g_ineq = eval_g0,
                eval_jac_g_ineq = eval_jac_g0, 
                w_now=w,
                opts = list("algorithm" = "NLOPT_LD_MMA",
                            "xtol_rel"=1.0e-8),
  )
  w <- out$solution
  # record
  w_chain <- rbind(w_chain, w)
  # step 2: check for rtermination terminate
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')

#2.3.3 Variance reduction
#What would you do it you wanted to reduce the variance of the stochastic gradient?
#(i)
m <- 1
eta <- 0.5
Tmax <- 500
kappa <- 100
Qstop <- 0 
t <- 0
#
#seeds
w_seed <- c(-0.3,3.0)
w <- w_seed
w_chain <- c()
cv_w <- w  
erf_fun <- function(w, z = z_obs, n=n_obs) {
  return( empirical_risk_fun(w, z, n) ) 
}
cv_grad_risk <- numDeriv::grad( erf_fun, cv_w ) #control variate
#
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  }
  else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * (numDeriv::grad( erf_fun, w ) -numDeriv::grad( erf_fun, cv_w ) +cv_grad_risk )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  # record
  w_chain <- rbind(w_chain, w)
  # control variate step
  if ( (t %% kappa) == 0) {
    cv_w <- w  
    erf_fun <- function(w, z = z_obs, n=n_obs) {
      return( empirical_risk_fun(w, z, n) ) 
    }
    cv_grad_risk <- numDeriv::grad( erf_fun, cv_w ) #controle variate
  }
  # step 2: check for termination 
  if ( t>= Tmax ) {
    Qstop <- 1
  } #not running?
}
plot(w_chain[,1], type='l') #+ abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') #+abline(h=w_true[2], col='red')

#(ii)
#
# The following is supposed to take veery long time to run. It is provided only for illustration purposes.  
# In the assesments the requested tasks will take less time.
#
m <- 1
eta <- 0.5
Tmax <- 500
kappa <- 10
Qstop <- 0 
t <- 0
#
#seeds
w_seed <- c(-0.3,3.0)
w <- w_seed
w_chain <- c()
cv_w <- w  
erf_fun <- function(w, z = z_obs, n=n_obs) {
  return( empirical_risk_fun(w, z, n) ) 
}
cv_grad_risk <- numDeriv::grad( erf_fun, cv_w ) #control variate
#
while ( Qstop == 0 ) {
  # counter
  t <- t +  1
  cat( t ) ; cat( ' ' ) ## counter added for display reasons
  # step 1: update  
  J <- sample.int(n = n_obs, size = m, replace = TRUE)
  if (m==1) {
    zbatch <- matrix(z_obs[J,],1,2)
  }
  else {
    zbatch <- z_obs[J,]
  }
  #eta <- learning_rate( t )
  erf_fun <- function(w, z = zbatch, n=m) {
    return( empirical_risk_fun(w, z, n) ) 
  }
  w <- w - eta * (numDeriv::grad( erf_fun, w ) -numDeriv::grad( erf_fun, cv_w ) +cv_grad_risk )
  #w <- w - eta * grad_risk_fun( w, zbatch, m )
  # record
  w_chain <- rbind(w_chain, w)
  # control variate step
  if ( (t %% kappa) == 0) {
    cv_w <- w  
    erf_fun <- function(w, z = z_obs, n=n_obs) {
      return( empirical_risk_fun(w, z, n) ) 
    }
    cv_grad_risk <- numDeriv::grad( erf_fun, cv_w ) #controle variate
  }
  # step 2: check for termination 
  if ( t>= Tmax ) {
    Qstop <- 1
  }
}
plot(w_chain[,1], type='l') #+abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') #+abline(h=w_true[2], col='red')

