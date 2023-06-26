include("delay_function.jl")
include("FSP_distribution.jl")

## theory result
function gamma_distribution_4state(α = 1, t0 = 1, k0 = 1, k1 = 1,  λ= 10)
      α = α
      θ = t0 / α
      r = 1
      N = 70
      p1 = [0, λ, 2*λ, λ, k0, k1]
      p2 = [α, θ, r]
      p3 = N
      p = [p1, p2, p3]
      tspan = (0.0, 500.0)
      saveat = [400]
      #saveat = [0 400]
      H = delay_mature_gamma
      model(du, u, p, t) = model_4state(du, u, p, t, H)
      prob1 = FSP_distr_4state(model, p, tspan, saveat)
      return (prob1)
end

prob = gamma_distribution_4state(4, 4, 0.5, 0.8, 20)
N = size(prob)[1] - 1
bins = collect(0:N);

plot(bins, prob,linewidth=5, size = (500,400),xlims = [0,40],
dpi=600, legend = :none,color = :blue,grid=0,
framestyle=:box,labelfontsize=14,tickfontsize = 12,)
xlabel!("mature mRNA#"); ylabel!("Probability")


## SSA result
using DelaySSAToolkit
using Random, Distributions
using StatsBase
using Interpolations
using Plots

function generate_data_mature(kon=0.5,koff=1,ρ2=20,ρ3=40,ρ4=20,r=1,α=4,mu=4)
    rates = [kon,kon,koff,koff,kon,koff,koff,kon,ρ2,ρ3,ρ4,r]
    reactant_stoich = [[1=>1],[2=>1],[3=>1],[4=>1],[4=>1],[3=>1],[2=>1],[1=>1],
    [2=>1],[3=>1],[4=>1],[6=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,3=>1],[3=>-1,4=>1],[4=>-1,1=>1],
    [4=>-1,3=>1],[3=>-1,2=>1],[2=>-1,1=>1],[1=>-1,4=>1],[5=>1],[5=>1],[5=>1],[6=>-1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)


    u0 = [1,0,0,0,0,0]
    de_chan0 = [[]]
    tf = 200000.
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
kon = 0.5; koff = 0.8; ρ2=20; ρ3=40; ρ4=20;
α=4; mu=4; r = 1;
jsol = generate_data_mature(kon,koff,ρ2,ρ3,ρ4,r,α,mu)

## interpolation
using Interpolations
tt = collect(range(0,200000,200001))
nodes = (jsol.t,)
mRNA = jsol[6,:]
mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
mRNA_expr = mRNA_itp(tt)
mRNA_prob = proportionmap(mRNA_expr)
scatter!(mRNA_prob,shape = [:circle],markersize=8)
