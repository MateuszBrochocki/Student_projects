library(actuar)

# prawdopodobieństwo porażki na podstawie 100 powtórzeń dla lognormalnej funkcji kosztów (jedn. poisson)
simulate_ruin_lnorm <- function(T, u, c, lambda, miu, sigma) {
  porazka<-0
  for (i in 1:100){
    t<- 0
    U <- u
    
    while (U >= 0) {
      przyrost<-rexp(1, rate = lambda)
      t <- t + przyrost
      if (t>T)
      {
        break
      }
      U <- U + c * przyrost - rlnorm(1, miu, sigma)
    }
    
    if(U < 0){
      porazka=porazka+1
      
    }
  }
  prawd<-porazka/100
  return(prawd)
}

# prawdopodobieństwo porażki na podstawie 100 powtórzeń dla rozkładu pareto funkcji kosztów (jedn. poisson)
simulate_ruin_pareto <- function(T, u, c, lambda, shape, scale) {
  porazka<-0
  for (i in 1:100){
    t<- 0
    U <- u
    
    while (t < T && U >= 0) {
      przyrost<-rexp(1, rate = lambda)
      t <- t + przyrost
      if (t>T)
      {
        break
      }
      U <- U + c * przyrost - rpareto(1, shape, scale)
    }
    
    if(U < 0){
      porazka=porazka+1
    }
  }
  prawd<-porazka/100
  return(prawd)
}

# prawdopodobieństwo porażki na podstawie 100 powtórzeń dla lognormalnej funkcji kosztów (niejedn. poisson)
simulate_ruin_nhpp_lnorm <- function(T, u, c, niejed, miu, sigma) {
  porazka<-0
  lambdaGwiazdka <- optimize(niejed, interval = c(0, T), maximum = TRUE)$objective
  for (i in 1:100){
    t <- 0
    U <- u
    while (U >= 0) {
      t_zero<-t
      # czas do kolejnej szkody (przerzedzanie)
      repeat {
        przyrost<-rexp(1, rate = lambdaGwiazdka)
        t <- t + przyrost
        if (t>T)
        {
          break
        }
        if (runif(1) <= niejed(t)/lambdaGwiazdka) break
      }
      if (t>T)
      {
        break
      }
      U <- U + c * (t-t_zero) - rlnorm(1, miu, sigma)
    }
    
    if(U < 0)
    {
      porazka<-porazka+1
    }
  }
  prawd<-porazka/100
  return(prawd)
}    


# prawdopodobieństwo porażki na podstawie 100 powtórzeń dla rozkładu pareto funkcji kosztów (niejedn. poisson)
simulate_ruin_nhpp_pareto <- function(T, u, c, niejed, shape, scale) {
  porazka<-0
  lambdaGwiazdka <- optimize(niejed, interval = c(0, T), maximum = TRUE)$objective
  for (i in 1:100){
    t <- 0
    U <- u
    while (U >= 0) {
      t_zero<-t
      # czas do kolejnej szkody (przerzedzanie)
      repeat {
        przyrost<-rexp(1, rate = lambdaGwiazdka)
        t <- t + przyrost
        if (t>T)
        {
          break
        }
        if (runif(1) <= niejed(t)/lambdaGwiazdka) break
      }
      if (t>T)
      {
        break
      }
      U <- U + c * (t-t_zero) - rpareto(1, shape, scale)
    }
    
    if(U < 0)
    {
      porazka<-porazka+1
    }
  }
  prawd<-porazka/100
  return(prawd)
}

# tutaj zmieniamy funkcję dla niejednorodnego procesu poissona
niejed<-function(t){
    
  cos(t)^2
    
}
#-------------------------------------------------------------------------------

# przykład jednorodny proces poissona
N<-100
prawd<-numeric(N)
Ti<-200
ost_prawd<-numeric(Ti)
g<-numeric(Ti)
d<-numeric(Ti)
alpha<-0.05
for (t in 1:Ti){
  for (i in 1:N){
    
    prawd[i]<-simulate_ruin_pareto(t, 3, 500, 1, 0.5, 1)
    
     
  }
  ost_prawd[t]<-mean(prawd)
  g[t]<-ost_prawd[t]+qt(1-alpha/2,df=N-1)*(sd(prawd)/sqrt(N))
  d[t]<-ost_prawd[t]-qt(1-alpha/2,df=N-1)*(sd(prawd)/sqrt(N))
}

plot(0:Ti, c(0,ost_prawd), type = "l", lwd=2,
     xlab = "Czas", ylab = "Prawdopodobienstwo ruiny", main = "Zależność prawdopodobieństwa ruiny od czasu")

# obszar
polygon(c(1:Ti, rev(1:Ti)), c(g, rev(d)),
        col = rgb(1,0,0,0.2), border = NA)  

   #-------------------------------------------------------------------------------

# przykład niejednorodny proces poissona
N<-100
prawd<-numeric(N)
Ti<-200
ost_prawd<-numeric(Ti)
g<-numeric(Ti)
d<-numeric(Ti)
alpha<-0.05
for (t in 1:Ti){
  for (i in 1:N){
    
    prawd[i]<-simulate_ruin_nhpp_lnorm(t, 5, exp(3)/4, niejed, 1, 2)

    
    
  }
  ost_prawd[t]<-mean(prawd)
  g[t]<-ost_prawd[t]+qt(1-alpha/2,df=N-1)*(sd(prawd)/sqrt(N))
  d[t]<-ost_prawd[t]-qt(1-alpha/2,df=N-1)*(sd(prawd)/sqrt(N))
}

plot(0:Ti, c(0,ost_prawd), type = "l", lwd=2,
     xlab = "Czas", ylab = "Prawdopodobienstwo ruiny", main = "Zależność prawdopodobieństwa ruiny od czasu")

# obszar
polygon(c(1:Ti, rev(1:Ti)), c(g, rev(d)),
        col = rgb(1,0,0,0.2), border = NA) 
   

