
using DelaySSAToolkit
using Distributions


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
    sol1 = solve(djprob, DelaySSAToolkit.SSAStepper())
    return(sol1)
end
