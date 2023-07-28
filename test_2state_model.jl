
##
using DelaySSAToolkit
using Random, Distributions
using StatsBase
using Interpolations
using Plots
include("generating_data.jl")


## generate data

function generate_data_mature(kon=0.5,koff=0.5,ρ=40,r=1,α=1,mu=1)
    rates = [kon,koff,ρ, r]
    reactant_stoich = [[1=>1],[2=>1],[2=>1],[4=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,1=>1],[3=>1],[4=>-1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)

    p0 = kon/(kon+koff)
    x  = rand(Binomial(1,p0),1)[1]

    u0 = [1-x,x,0,0]
    de_chan0 = [[]]
    tf = 200000.
    tspan = (0,tf)
    dprob = DiscreteProblem(u0, tspan)
    delay_trigger_affect! = function (integrator, rng)
        alpha = α;
        τ=rand(Gamma(alpha,mu/alpha))
        append!(integrator.de_chan[1], τ)
    end
    delay_trigger = Dict(3=>delay_trigger_affect!)
    delay_complete = Dict(1=>[3=>-1,4=>1])
    delay_interrupt = Dict()
    delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)

    djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
     de_chan0, save_positions=(true,true))
    sol1 = solve(djprob, SSAStepper())
    return(sol1)
end


kon = 0.5; koff = 0.8; rou = 40;
alpha0 = 5; mu = 2;r = 1;
N = 50000
mRNA_mature= zeros(N)
jsol = generate_data_mature(kon,koff,rou,r,alpha0,mu)

##

using Interpolations
tt = collect(range(0,200000,200001))
nodes = (jsol.t,)
mRNA = jsol[4,:]
promoter = jsol[1,:]
mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
promoter_itp = Interpolations.interpolate(nodes,promoter, Gridded(Constant{Previous}()))

mRNA_expr = mRNA_itp(tt)
promoter_expr = promoter_itp(tt)

mRNA_prob = proportionmap(mRNA_expr[promoter_expr.==0])
scatter(mRNA_prob,shape = [:circle],markersize=8)




##

##

include("delay_function.jl")
include("FSP_distribution.jl")


k0=0.5;k1=0.8;λ1 = 40;λ2 = 0;
L=5; t0=2;r=1
H = delay_mature_gamma;
prob = mature_mRNA_distribution(H,L,t0,r,k0,k1,λ1,λ2)
N = length(prob)-1
bins = collect(0:N)

p0 = k0/(k1+k0)

plot!(bins, prob,linewidth=5, size = (500,400),xlims = [0,60],
dpi=600, legend = :none,grid=0,
framestyle=:box,labelfontsize=14,tickfontsize = 12,)
xlabel!("mature mRNA#"); ylabel!("Probability")
