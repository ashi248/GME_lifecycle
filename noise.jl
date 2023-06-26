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
     du[1] = λ1*SS*k0/(k0+k1)+k0*u[2]-k1*u[1];
     du[2] = λ2*SS*k1/(k0+k1)+k1*u[1]-k0*u[2];
     du[3] = 2*λ1*SS*u[1]+k0*u[4]-k1*u[3];
     du[4] = 2*λ2*SS*u[2]+k1*u[3]-k0*u[4];
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
      Mean = sol[1] + sol[2]
      Var = sol[3] + sol[4] + Mean - Mean^2
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


##
function general_distribution_noise(p1,p2)
      H = delay_mixture_gamma
      p = [p1,p2]
      model(du,u,p,t) = model_2state_noise(du,u,p,t,H)

      u0 = [0;0;0;0]
      tspan = (0.0,200.0);
      saveat = [200]
      moments = ODEProblem(model, u0, tspan,p);
      t = saveat
      sol = Array(solve(moments, Tsit5(), u0=u0, saveat=t));
      Mean = sol[1] + sol[2]
      Var = sol[3] + sol[4] + Mean - Mean^2
      return([Mean,Var])
end


##
function model_2state_noise(du,u,p,t,H)
     λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4];
     p2 = p[2] # α and θ

     SS = H(t,p2)[1];
     du[1] = λ1*SS*k0/(k0+k1)+k0*u[2]-k1*u[1];
     du[2] = λ2*SS*k1/(k0+k1)+k1*u[1]-k0*u[2];
     du[3] = 2*λ1*SS*u[1]+k0*u[4]-k1*u[3];
     du[4] = 2*λ2*SS*u[2]+k1*u[3]-k0*u[4];
end

##
function gamma_distribution_noise(H,p1,p2)
     p = [p1,p2]
      model(du,u,p,t) = model_2state_noise(du,u,p,t,H)

      u0 = [0;0;0;0]
      tspan = (0.0,50.0);
      saveat = [50]
      moments = ODEProblem(model, u0, tspan,p);
      t = saveat
      sol = Array(solve(moments, Tsit5(), u0=u0, saveat=t));
      Mean = sol[1] + sol[2]
      Var = sol[3] + sol[4] + Mean - Mean^2
      return([Mean,Var])
end

##
function gamma_distribution_noise_time(saveat,H,p1,p2)

      p = [p1,p2]
      model(du,u,p,t) = model_2state_noise(du,u,p,t,H)

      u0 = [0;0;0;0]
      tspan = (0.0,50.0);
      saveat = saveat
      moments = ODEProblem(model, u0, tspan,p);
      sol = Array(solve(moments, Tsit5(), u0=u0, saveat=saveat));

      Mean = zeros(length(saveat))
      Var = zeros(length(saveat))

      for i in 1:length(saveat)
            Mean[i] = sol[1,i] + sol[2,i]
            Var[i] = sol[3,i] + sol[4,i] + Mean[i] - Mean[i]^2
      end

      return([Mean,Var])
end
