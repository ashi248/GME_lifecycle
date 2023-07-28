## SSA result
using DelaySSAToolkit
using Random, Distributions
using StatsBase
using Interpolations
using Plots
using JLD2

function generate_data_mature(state = 1,ρ2=20,ρ3=40,ρ4=20,r=1,α=4,mu=4)
    k12 = 0.1;k23=0.5;k34=0.5;k41=0.5;
    k43 = 0.5;k32=0.5;k21=0.5;k14 =0.5
    rates = [k12,k23,k34,k41,k43,k32,k21,k14,ρ2,ρ3,ρ4,r]
    reactant_stoich = [[1=>1],[2=>1],[3=>1],[4=>1],[4=>1],[3=>1],[2=>1],[1=>1],
    [2=>1],[3=>1],[4=>1],[6=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,4=>1],[4=>-1,1=>1],
    [4=>-1,3=>1],[3=>-1,2=>1],[2=>-1,1=>1],[1=>-1,4=>1],[5=>1],[5=>1],[5=>1],[6=>-1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)


    u0 = zeros(6)
    u0[state] = 1
    de_chan0 = [[]]
    tf = 50.
    tspan = (0,tf)
    dprob = DiscreteProblem(u0, tspan)
    delay_trigger_affect! = function (integrator, rng)
        alpha = α;
        τ=rand(Gamma(alpha,mu/alpha))
        append!(integrator.de_chan[1], τ)
    end
    delay_trigger = Dict(9=>delay_trigger_affect!,10=>delay_trigger_affect!,11=>delay_trigger_affect!)
    delay_complete = Dict(1=>[5=>-1,6=>1])
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)

    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
     de_chan0, save_positions=(true,true))
    sol1 = solve(djprob, SSAStepper())
    return(sol1)
end

## generate data
ρ2=10; ρ3=20; ρ4=10;
α=10; mu=4; r = 1;
N = 400000
tt = collect(range(0,50,51))

##
# state = 1
state = 1
mRNA_expr_1 = zeros(N,length(tt))
for i in 1:N
  print(i)
  jsol = generate_data_mature(state,ρ2,ρ3,ρ4,r,α,mu)
  nodes = (jsol.t,)
  mRNA = jsol[6,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA_expr_1[i,:] = mRNA_itp(tt)
end

state = 2
mRNA_expr_2 = zeros(N,length(tt))
for i in 1:N
  print(i)
  jsol = generate_data_mature(state,ρ2,ρ3,ρ4,r,α,mu)
  nodes = (jsol.t,)
  mRNA = jsol[6,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA_expr_2[i,:] = mRNA_itp(tt)
end

state = 3
mRNA_expr_3 = zeros(N,length(tt))
for i in 1:N
  print(i)
  jsol = generate_data_mature(state,ρ2,ρ3,ρ4,r,α,mu)
  nodes = (jsol.t,)
  mRNA = jsol[6,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA_expr_3[i,:] = mRNA_itp(tt)
end

state = 4
mRNA_expr_4 = zeros(N,length(tt))
for i in 1:N
  print(i)
  jsol = generate_data_mature(state,ρ2,ρ3,ρ4,r,α,mu)
  nodes = (jsol.t,)
  mRNA = jsol[6,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA_expr_4[i,:] = mRNA_itp(tt)
end

save_object("data/four_state_model_SSA.jld2",(mRNA_expr_1,mRNA_expr_2,
mRNA_expr_3,mRNA_expr_4))


##
include("delay_function.jl")
include("FSP_distribution.jl")
using Distributions, Distances
using DelimitedFiles, Plots
## theory result
function gamma_distribution_4general(α = 1, t0 = 1, λ= 10)
      α = α
      θ = t0 / α
      r = 1
      N = 70
      k12 = 0.1;k23=0.5;k34=0.5;k41=0.5;
      k43 = 0.5;k32 = 0.5;k21=0.5;k14 = 0.5
      p1 = [0, λ, 2*λ, λ, k12, k23,k34,k41,k43,k32,k21,k14]
      p2 = [α, θ, r]
      p3 = N
      p = [p1, p2, p3]
      tspan = (0.0, 800.0)
      saveat = collect(range(0,50,51))
      #saveat = [0 400]
      H = delay_mature_gamma
      model(du, u, p, t) = model_4general(du, u, p, t, H)
      prob1,prob2,prob3,prob4 = FSP_distr_4general(model, p, tspan, saveat)
      return ([prob1,prob2,prob3,prob4])
end
prob1,prob2,prob3,prob4 = gamma_distribution_4general(10, 4, 10)




###
## state 1

time0 = 5
N = size(prob1)[1] - 1
bins = collect(0:N);

f11 = Plots.plot(bins, prob1[:,time0+1],label = "model",xlims = [0,20],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_1 = proportionmap(mRNA_expr_1[:,time0+1])
scatter!(mRNA_prob_1,shape = [:circle],label="SSA")

time0 = 10
f12 = Plots.plot(bins, prob1[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_1 = proportionmap(mRNA_expr_1[:,time0+1])
scatter!(mRNA_prob_1,shape = [:circle],label="SSA")

time0 = 50
f13 = Plots.plot(bins, prob1[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_1 = proportionmap(mRNA_expr_1[:,time0+1])
scatter!(mRNA_prob_1,shape = [:circle],label="SSA")

###
## state2
time0 = 5
N = size(prob2)[1] - 1
bins = collect(0:N);

f21 = Plots.plot(bins, prob2[:,time0+1],label = "model",xlims = [0,20],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_2 = proportionmap(mRNA_expr_2[:,time0+1])
scatter!(mRNA_prob_2,shape = [:circle],label="SSA")


time0 = 10
N = size(prob2)[1] - 1
bins = collect(0:N);

f22 = Plots.plot(bins, prob2[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_2 = proportionmap(mRNA_expr_2[:,time0+1])
scatter!(mRNA_prob_2,shape = [:circle],label="SSA")


time0 = 50
N = size(prob2)[1] - 1
bins = collect(0:N);

f23 = Plots.plot(bins, prob2[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_2 = proportionmap(mRNA_expr_2[:,time0+1])
scatter!(mRNA_prob_2,shape = [:circle],label="SSA")

###
## state 3
time0 = 5
N = size(prob3)[1] - 1
bins = collect(0:N);

f31 = Plots.plot(bins, prob3[:,time0+1],label = "model",xlims = [0,20],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_3 = proportionmap(mRNA_expr_3[:,time0+1])
scatter!(mRNA_prob_3,shape = [:circle],label="SSA")


time0 = 10
f32 = Plots.plot(bins, prob3[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_3 = proportionmap(mRNA_expr_3[:,time0+1])
scatter!(mRNA_prob_3,shape = [:circle],label="SSA")


time0 = 50
f33 = Plots.plot(bins, prob3[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_3 = proportionmap(mRNA_expr_3[:,time0+1])
scatter!(mRNA_prob_3,shape = [:circle],label="SSA")


##state 4

time0 = 5
N = size(prob4)[1] - 1
bins = collect(0:N);

f41 = Plots.plot(bins, prob4[:,time0+1],label = "model",xlims = [0,20],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_4 = proportionmap(mRNA_expr_4[:,time0+1])
scatter!(mRNA_prob_4,shape = [:circle],label="SSA")


time0 = 10
f42 = Plots.plot(bins, prob4[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_4 = proportionmap(mRNA_expr_4[:,time0+1])
scatter!(mRNA_prob_4,shape = [:circle],label="SSA")


time0 = 50
f43 = Plots.plot(bins, prob4[:,time0+1],label = "model",xlims = [0,30],
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box)
mRNA_prob_4 = proportionmap(mRNA_expr_4[:,time0+1])
scatter!(mRNA_prob_4,shape = [:circle],label="SSA")
##



plot(f11,f21,f31,f41,f12,f22,f32,f42,f13,f23,f33,f43,layout=(3,4),linewidth=4,dpi=800,
size = (1000,600),tickfontsize = 10)

Plots.savefig("figure0/4state_total.png")
