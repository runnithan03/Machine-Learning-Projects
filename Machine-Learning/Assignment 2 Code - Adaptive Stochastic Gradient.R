#(4)
data_generating_model <- function(n,w) {
  z <- rep( NaN, times=n*3 )
  z <- matrix(z, nrow = n, ncol = 3)
  z[,1] <- rep(1,times=n)
  z[,2] <- runif(n, min = -10, max = 10)
  p <- w[1]*z[,1] + w[2]*z[,2] 
  p <- exp(p) / (1+exp(p))
  z[,3] <- rbinom(n, size = 1, prob = p)
  ind <- (z[,3]==0)
  z[ind,3] <- -1
  x <- z[,1:2]
  y <- z[,3]
  return(list(z=z, x=x, y=y))
}
n_obs <- 1000000
w_true <- c(-3,4)
set.seed(2023)
out <- data_generating_model(n = n_obs, w = w_true)
set.seed(0)
z_obs <- out$z #z=(x,y)
x <- out$x
y <- out$y
z_obs2=z_obs
z_obs2[z_obs[,3]==-1,3]=0
w_true <- as.numeric(glm(z_obs2[,3]~ 1+ z_obs2[,2],family = "binomial")$coefficients)
w_true

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

loss_fun <- function(w,z) {
  x = z[1]
  y = z[2]
  lambda <- 0 
  ell <- max(0, 1-y * t(w) * x) + lambda*norm(w, type = c("2"))^2
  return (ell)
}

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
which.min(w_chain) #511
w_chain[,511] #-0.3364010976 -0.5445853730

#(5)
data_generating_model <- function(n,w) {
  z <- rep( NaN, times=n*3 )
  z <- matrix(z, nrow = n, ncol = 3)
  z[,1] <- rep(1,times=n)
  z[,2] <- runif(n, min = -10, max = 10)
  p <- w[1]*z[,1] + w[2]*z[,2] p <- exp(p) / (1+exp(p))
  z[,3] <- rbinom(n, size = 1, prob = p)
  ind <- (z[,3]==0)
  z[ind,3] <- -1
  x <- z[,1:2]
  y <- z[,3]
  return(list(z=z, x=x, y=y))
}
n_obs <- 1000000
w_true <- c(-3,4)
set.seed(2023)
out <- data_generating_model(n = n_obs, w = w_true)
set.seed(0)
z_obs <- out$z #z=(x,y)
x <- out$x
y <- out$y
z_obs2=z_obs
z_obs2[z_obs[,3]==-1,3]=0
w_true <- as.numeric(glm(z_obs2[,3]~ 1+ z_obs2[,2],family = "binomial")$coefficients)
w_true

#Adagrad
m <- 1
eta <- 1.0
Tmax <- 500
w_seed <- c(0,0)
w <- w_seed
w_chain <- c()
Qstop <- 0 
t <- 0
G <- rep(0.0,times=length(w))
eps <- 10^(-6)

loss_fun <- function(w,z) {
  x = z[1]
  y = z[2]
  lambda <- 0 
  ell <- max(0, 1-y * t(w) * x) + lambda*norm(w, type = c("2"))^2
  return (ell)
}

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
which.min(w_chain) 
w_chain[1,] 

plot(w_chain[,1], type='l') +
  abline(h=w_true[1], col='red')
plot(w_chain[,2], type='l') +
  abline(h=w_true[2], col='red')

1-pchisq(1.103,1)
