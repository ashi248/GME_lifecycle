using Distributions, Distances
using DelimitedFiles
using SpecialFunctions
using QuadGK

##
# transform markov chain to rate matrix
function path_matrix(N,p)
      k10 = p[1]; k01 = p[2]; v = p[3]; d = p[4];

      T = zeros(N+1,N+1)
      T[N+1,N+1] = -k01; T[N+1,1] = k01;

      if N>=2
            T[1,1] = -(k10+v+d); T[1,2] = v; T[1,N+1] = k10;
      else
            T[1,1] = -(k10+v+d); T[1,N+1] = k10;
      end

      if N>=3
            for i in 2:(N-1)
                  T[i,i] = -(v+d)
                  T[i,i+1] = v
            end
      elseif N==2
            T[2,2] = -(v+d)
      end

      if N>=3
            T[N,N] = -(v+d)
      end
      return(T)
end

##
# survival function and delay function for general pathway
function delay_nascent_matrix(x,p)
      T = p[1]
      N = size(T,1)
      tau = zeros(1,N )
      tau[1] = 1
      h0 = tau*exp(T*x)*ones(N,1)
      return(h0[1])
end

function density_nascent_matrix(x,T)
      N = size(T,1)
      tau = zeros(1,N )
      tau[1] = 1
      t0 = -T*ones(N,1)
      h = tau*exp(T*x)*t0
      return(h[1])
end

function delay_mature_matrix(t,p)
      T = p[1]
      r = p[2]
      S(u) = density_nascent_matrix(u,T)*exp(-r*(t-u))
      if t==0
            h0=0
      else h0 = quadgk(S,0,t)
      end
      return(h0[1])
end


function delay_mature_matrix_total(t,p)
      T = p[1]
      r = p[2]
      α2 = p[3]
      θ2 = p[4]
      # S(u) = density_nascent_matrix(u,T)*exp(-r*(t-u))
      #S(u) = density_nascent_matrix(u,T)*gamma(α2, (t-u)/θ2)/gamma(α2)
      S(u) = density_nascent_matrix(u,T)*(1-cdf(Gamma(α2,θ2),(t-u)))
      if t==0
            h0=0
      else h0 = quadgk(S,0,t)
      end
      return(h0[1])
end


##  delay function for gamma distribution

function delay_nascent_gamma(x,p)
    α = p[1]
    θ = p[2]
    h = 1-cdf(Gamma(α,θ),x)
    return(h)
end

function delay_lognormal_gamma(x,p)
    μ = p[1]
    σ = p[2]
    h = 1-cdf(LogNormal(μ, σ),x)
    return(h)
end


function density_nascent_gamma(x,p)
      α = p[1]
      θ = p[2]
      h0 = pdf(Gamma(α,θ),x)
      return(h0)
end

function delay_mature_gamma(t,p)
      α = p[1]
      θ1 = p[2]
      θ2 = p[3]
      S(u) = pdf(Gamma(α,θ1),u)*exp(-(t-u)/θ2)
      if t==0
            h0=0
      else h0 = quadgk(S,0,t)
      end
      return(h0[1])
end


function delay_mature_total(t,p)
      α1 = p[1]
      θ1 = p[2]
      α2 = p[3]
      θ2 = p[4]
      #S(u) = pdf(Gamma(α1,θ1),u)*(1-cdf(Gamma(α2,θ2),(t-u)))
      S(u) = pdf(Gamma(α1,θ1),u)*gamma(α2, (t-u)/θ2)/gamma(α2)
      if t==0
            h0=0
      else h0 = quadgk(S,0,t)
      end
      return(h0[1])
end


function delay_mature_total_log(t,p)
      μ = p[1]
      σ = p[2]
      α2 = p[3]
      θ2 = p[4]
      #S(u) = pdf(Gamma(α1,θ1),u)*(1-cdf(Gamma(α2,θ2),(t-u)))
      S(u) = pdf(LogNormal(μ, σ),u)*gamma(α2, (t-u)/θ2)/gamma(α2)
      if t==0
            h0=0
      else h0 = quadgk(S,0,t)
      end
      return(h0[1])
end
