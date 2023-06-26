using DelaySSAToolkit
using Random, Distributions
#using Catalyst, Plots, DiffEqJump
using StatsBase
using Interpolations
using DataFrames
using JLD2

function generate_data(kon=0.5,koff=0.5,ρ=40,r=1,α=0.1,mu=2)
    # kon,koff,ρ, r = [0.5, 0.5, 40, 1]
    # α=0.1;
    # mu=2
    rates = [kon,koff,ρ, r]
    reactant_stoich = [[1=>1],[2=>1],[2=>1],[4=>1]]
    net_stoich = [[1=>-1,2=>1],[2=>-1,1=>1],[3=>1],[4=>-1]]
    mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
    jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)

    p0 = kon/(kon+koff)
    x  = rand(Binomial(1,p0),1)[1]

    u0 = [1-x,x,0,0]
    de_chan0 = [[]]
    tf = 100.
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
    nuc = sol1[3,length(sol1.t)]
    cyt = sol1[4,length(sol1.t)]
    return(nuc,cyt)
end

M0 = 50
Kon = rand(LogUniform(0.1,2),M0)
Koff = rand(LogUniform(0.1,2),M0)
fon = Kon./(Kon .+ Koff)
rou = rand(LogUniform(10,100),M0)
alpha0 = rand(LogUniform(0.1,20),M0)
Mu = rand(LogUniform(0.1,10),M0)
Parameter = DataFrame(kon = Kon,koff = Koff,rou = rou,
alpha = alpha0,mu = Mu)


N = 100
mRNA_nuc = zeros(N,M0)
mRNA_cyt = zeros(N,M0)
for j in 1:M0
     print(j)
    for i in 1:N
        nuc0,cyt0 = generate_data(Kon[j],Koff[j],rou[j],1,alpha0[j],Mu[j])
        mRNA_nuc[i,j] = nuc0
        mRNA_cyt[i,j] = cyt0
    end
end
