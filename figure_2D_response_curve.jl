using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots
using QuadGK
include("FSP_distribution.jl")
include("noise.jl")
include("delay_function.jl")


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

function gamma_distribution_noise_time(saveat,H,p1,p2)

      p = [p1,p2]
      model(du,u,p,t) = model_2state_noise(du,u,p,t,H)

      u0 = [0;1;0;0]
      tspan = (0.0,200.0);
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


using LaTeXStrings

#
fig = Plots.plot(xlabel = "Time",
xlims = [0,8],ylabel = "Mean of mature RNA",
labelfontsize=14,dpi=600,legendfontsize = 10,size = [500,400],
grid=0,framestyle=:box,legend=:bottomright,tickfontsize = 12)

p1  = [20,0,10,0]    #[λ1,λ2,k0,k1]
α1= 1/0.1;θ1 = 2/α1; θ2 = 1
p2 = [α1,θ1,θ2]
H = delay_mature_gamma
saveat = collect(range(0,50,200))
Mean1,Var1 = gamma_distribution_noise_time(saveat,H,p1,p2)
normal_mean1 =  Mean1./θ2
Plots.plot!(saveat, normal_mean1,legendfontsize = 14,linewidth=4,
label = L"\eta_\tau^2=0.1")



α1= 1/1;θ1 = 2/α1; θ2 = 1
p2 = [α1,θ1,θ2]
H = delay_mature_gamma
saveat = collect(range(0,50,200))
Mean1,Var1 = gamma_distribution_noise_time(saveat,H,p1,p2)
normal_mean1 =  Mean1./θ2
Plots.plot!(saveat, normal_mean1,linewidth=4,
label = L"\eta_\tau^2=1")


α1= 1/10;θ1 = 2/α1; θ2 = 1
p2 = [α1,θ1,θ2]
H = delay_mature_gamma
saveat = collect(range(0,50,200))
Mean1,Var1 = gamma_distribution_noise_time(saveat,H,p1,p2)
normal_mean1 =  Mean1./θ2
Plots.plot!(saveat, normal_mean1,linewidth=4,
label = L"\eta_\tau^2=10")

savefig("figure0/delay_to_response.png")
