M=1000
N=10000
X <- matrix(rnorm(N*M, 0, 1), nrow=N, ncol=M)

S_T <- function(x) { return(110*exp((0.05-0.3**2/2)*T + 0.3*sqrt(1)*x))}
X <- sapply(X, S_T)

f<- function(x) { return(x*max((x-110),0)) }
X <- sapply(X, f)
X <- matrix(X, nrow=N, ncol=M)
mu_hats <- apply(X, MARGIN=2, FUN=mean)

hist(X, breaks="Scott")#, density=True)

hist(mu_hats, breaks="Scott", freq=F)


mu_mu = mean(mu_hats)
sig_mu = sqrt(var(mu_hats))
grid = seq(2550, 2900, by=(2900-2550)/200)
lines(grid, dnorm(grid, mu_mu, sig_mu), col='red')
