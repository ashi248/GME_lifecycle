using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles
using DSP
using SpecialFunctions
using DiffEqSensitivity, DifferentialEquations
using LinearAlgebra
using StatsBase


##
function model_2state(du, u, p, t)
      α = p[1]
      θ = 1 / α
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4]
      N = convert(Int,p[5])
      SS = gamma(α, t/θ)/gamma(α)
      #SS = exp(-t)
      du[1] = -λ1*SS*u[1]+k1*u[2]-k1*u[1]
      du[2] = -λ2*SS*u[2]+k0*u[1]-k0*u[2]
      for i = 1:N
            du[2*i+1] =λ1*SS*(u[2*(i-1)+1]-u[2*i+1]) + k1*u[2*i+2]-k1*u[2*i+1]
            du[2*i+2] =λ2*SS*(u[2*(i-1)+2]-u[2*i+2]) - k0*u[2*i+2]+k0*u[2*i+1]
      end
end


function FSP_distr(model, p, tspan, saveat)
      α = p[1]
      θ = 1 / α
      r = 1
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4] # number of product
      N = convert(Int,p[5])
      u0 = [1; 1; zeros(2 * N)]
      prob = ODEProblem(model, u0, tspan, p)
      # create data
      t = saveat
      sol = Array(solve(prob, Tsit5(), u0 = u0, saveat = t))
      prob_mRNA = sol[collect(1:2:(2*N+2)), :]
      data = convert(Array, prob_mRNA)
      return (data)
end


function fit_function_G2(p::NamedTuple{(:α, :λ1, :k0, :k1)},numb::Int)
      tspan = (0.0, 200.0)
      saveat = [200]
      p0 = [exp(p.α), exp(p.λ1), exp(p.k0), exp(p.k1),numb]
      model(du, u, p, t) = model_2state(du, u, p0, t)
      prob0 = FSP_distr(model, p0, tspan, saveat)
      prob = DSP.conv(prob0,prob0)
      return (prob)
end

function fit_function(p::NamedTuple{(:α, :λ1, :k0, :k1)},numb::Int)
      tspan = (0.0, 200.0)
      saveat = [200]
      p0 = [exp(p.α), exp(p.λ1), exp(p.k0), exp(p.k1),numb]
      model(du, u, p, t) = model_2state(du, u, p0, t)
      prob = FSP_distr(model, p0, tspan, saveat)
      return (prob)
end

##
function Log_Likelihood_nascent(data,p)

      number = convert.(Int,data.keys)
      M = maximum(number)
      p0 = [exp.(p);M]
      tspan = (0.0, 200.0)
      saveat = [200]
      model(du, u, p, t) = model_2state(du, u, p0, t)
      prob = FSP_distr(model, p0, tspan, saveat)

      prob = abs.(prob)
      prob0 = [prob[i+1] for i in number]
      tot_loss = -sum(data.vals.* log.(prob0))
      return (tot_loss)
end

##
function model_2state_markov(du, u, p, t)
      λ1 = p[1]
      λ2 = 0
      k0 = p[2]
      k1 = p[3]
      N = convert(Int,p[4])
      SS = exp(-t)
      du[1] = -λ1 * SS * u[1] + k1 * u[2] - k1 * u[1]
      du[2] = -λ2 * SS * u[2] + k0 * u[1] - k0 * u[2]
      for i = 1:N
            du[2*i+1] =
                  λ1 * SS * (u[2*(i-1)+1] - u[2*i+1]) + k1 * u[2*i+2] -
                  k1 * u[2*i+1]
            du[2*i+2] =
                  λ2 * SS * (u[2*(i-1)+2] - u[2*i+2]) - k0 * u[2*i+2] +
                  k0 * u[2*i+1]
      end
end


function FSP_distr_markov(model, p, tspan, saveat)
      λ1 = p[1]
      λ2 = 0
      k0 = p[2]
      k1 = p[3] # number of product
      N = convert(Int,p[4])
      u0 = [1; 1; zeros(2 * N)]
      prob = ODEProblem(model, u0, tspan, p)
      # create data
      t = saveat
      sol = Array(solve(prob, Tsit5(), u0 = u0, saveat = t))
      prob_mRNA = sol[collect(1:2:(2*N+2)), :]
      data = convert(Array, prob_mRNA)
      return (data)
end


function Log_Likelihood_nascent_markov(data,p)
      number = convert.(Int,data.keys)
      M = maximum(number)
      p0 = [exp.(p);M]
      tspan = (0.0, 200.0)
      saveat = [200]
      model(du, u, p, t) = model_2state_markov(du, u, p0, t)
      prob = FSP_distr_markov(model, p0, tspan, saveat)

      prob = abs.(prob)
      prob0 = [prob[i+1] for i in number]
      tot_loss = -sum(data.vals.* log.(prob0))
      return (tot_loss)
end

##


function model_2state_fix(du, u, p, t)
      λ1 = p[1]
      λ2 = 0
      k0 = p[2]
      k1 = p[3]
      N = convert(Int,p[4])
      SS = ifelse(t>=1,0,1)

      du[1] = -λ1 * SS * u[1] + k1 * u[2] - k1 * u[1]
      du[2] = -λ2 * SS * u[2] + k0 * u[1] - k0 * u[2]
      for i = 1:N
            du[2*i+1] =
                  λ1 * SS * (u[2*(i-1)+1] - u[2*i+1]) + k1 * u[2*i+2] -
                  k1 * u[2*i+1]
            du[2*i+2] =
                  λ2 * SS * (u[2*(i-1)+2] - u[2*i+2]) - k0 * u[2*i+2] +
                  k0 * u[2*i+1]
      end
end


function FSP_distr_fix(model, p, tspan, saveat)
      λ1 = p[1]
      λ2 = 0
      k0 = p[2]
      k1 = p[3] # number of product
      N = convert(Int,p[4])
      u0 = [1; 1; zeros(2 * N)]
      prob = ODEProblem(model, u0, tspan, p)
      # create data
      t = saveat
      sol = Array(solve(prob, Tsit5(), u0 = u0, saveat = t))
      prob_mRNA = sol[collect(1:2:(2*N+2)), :]
      data = convert(Array, prob_mRNA)
      return (data)
end


function Log_Likelihood_nascent_fix(data,p)
      number = convert.(Int,data.keys)
      M = maximum(number)
      p0 = [exp.(p);M]
      tspan = (0.0, 200.0)
      saveat = [200]
      model(du, u, p, t) = model_2state_fix(du, u, p0, t)
      prob = FSP_distr_fix(model, p0, tspan, saveat)

      prob = abs.(prob)
      prob0 = [prob[i+1] for i in number]
      tot_loss = -sum(data.vals.* log.(prob0))
      return (tot_loss)
end
