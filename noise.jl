using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots
using QuadGK
include("delay_function.jl")
include("FSP_distribution.jl")


function prob2noise(prob)
      N = length(prob)-1
      number = collect(0:N)
      Mean = sum(number.*prob)
      var = sum(number.^2 .*prob) - Mean^2
      Noise = var/Mean^2
      return([Mean,Noise])
end

##
function noise_formula((u,v),p)
    H = delay_mature_gamma
    f = H(u,p)*H(v,p)*exp(-(k0+k1)*abs(u-v))
    return(f)
end

##
function model_2state_noise(du,u,p,t,H)
     λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4];
     p2 = p[2] # α and θ

     SS = H(t,p2)[1];
     du[1] = λ1*SS+k1*u[2]-k1*u[1];
     du[2] = λ2*SS+k0*u[1]-k0*u[2];
     du[3] = 2*λ1*SS*u[1]+k1*u[4]-k1*u[3];
     du[4] = 2*λ2*SS*u[2]+k0*u[3]-k0*u[4];
end


##
function gamma_distribution_noise(H,p1,p2)

      p = [p1,p2]
      model(du,u,p,t) = model_2state_noise(du,u,p,t,H)

      u0 = [0;0;0;0]
      tspan = (0.0,400.0);
      saveat = [400]
      moments = ODEProblem(model, u0, tspan,p);
      t = saveat
      sol = Array(solve(moments, Tsit5(), u0=u0, saveat=t));
      Mean = sol[1]
      Var = sol[3] + Mean - Mean^2
      return([Mean,Var])
end


##

function gini_coef(p)
    f1(x) = delay_mature_gamma(x,p)
    f2(x) = delay_mature_gamma(x,p)^2
    h1  = quadgk(f1,0,Inf)[1]
    h2 = quadgk(f2,0,Inf)[1]
    GC = 1 - h2/h1
    return(GC)
end
