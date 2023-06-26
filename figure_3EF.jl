using Catalyst
using DelaySSAToolkit
using Plots
using Distributions
using Random, Distributions

kon,koff,ρ, r = [0.5, 0.5, 40, 1]
rates = [kon,koff,ρ, r]
reactant_stoich = [[1=>1],[2=>1],[2=>1],[4=>1]]
net_stoich = [[1=>-1,2=>1],[2=>-1,1=>1],[3=>1],[4=>-1]]
mass_jump = DelaySSAToolkit.MassActionJump(rates, reactant_stoich, net_stoich; scale_rates =false)
jumpset = DelaySSAToolkit.JumpSet((),(),nothing,mass_jump)


u0 = [0,1,0,0]
de_chan0 = [[]]
tf = 10000.
tspan = (0,tf)
dprob = DiscreteProblem(u0, tspan)


delay_trigger_affect! = function (integrator, rng)
    alpha = 0.1;
    τ=rand(Gamma(alpha,2/alpha))
    append!(integrator.de_chan[1], τ)
end
delay_trigger = Dict(3=>delay_trigger_affect!)
delay_complete = Dict(1=>[3=>-1,4=>1])
delay_interrupt = Dict()
delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)

djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
 de_chan0, save_positions=(true,true))
sol1 = solve(djprob, SSAStepper())



 delay_trigger_affect! = function (integrator, rng)
     alpha = 1;
     τ=rand(Gamma(alpha,2/alpha))
     append!(integrator.de_chan[1], τ)
 end
 delay_trigger = Dict(3=>delay_trigger_affect!)
 delay_complete = Dict(1=>[3=>-1,4=>1])
 delay_interrupt = Dict()
 delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)
 djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
  de_chan0, save_positions=(true,true))
 sol2 = solve(djprob, SSAStepper())



delay_trigger_affect! = function (integrator, rng)
     alpha = 10;
     τ=rand(Gamma(alpha,2/alpha))
     append!(integrator.de_chan[1], τ)
 end
 delay_trigger = Dict(3=>delay_trigger_affect!)
 delay_complete = Dict(1=>[3=>-1,4=>1])
 delay_interrupt = Dict()
 delayjumpset = DelayJumpSet(delay_trigger,delay_complete,delay_interrupt)

 djprob = DelayJumpProblem(dprob, DelayRejection(), jumpset, delayjumpset,
  de_chan0, save_positions=(true,true))
 sol3 = solve(djprob, SSAStepper())



##
using LaTeXStrings
using Interpolations

tt = collect(range(0,100,2000))
sol = sol1
nodes = (sol.t,)
expr0 = sol[2,:]
itp_fun = Interpolations.interpolate(nodes,expr0, Gridded(Constant{Previous}()))
promoter= itp_fun(tt)

 fig1 = plot(tt,promoter,linewidth=4,legend = :none,
 ylabel = "Promoter",xlabel = "Time", color = "black", xlims = [0,80],
 grid=0,labelfontsize=14,tickfontsize=10,yticks = [0,1],dpi=600)

 fig3 = plot(sol1.t,sol1[4,:],linewidth=4,label = L"\eta_\tau^2=10",
  color = "blue",alpha = 0.8,grid=0,framestyle=:box,xlims = [0,80],
  labelfontsize=14,legendfontsize=14,tickfontsize=10,dpi=600,
   ylabel = "Mature RNA")
 plot!(sol2.t,sol2[4,:],linewidth=4,label = L"\eta_\tau^2=1",color = "red")
 plot!(sol3.t,sol3[4,:],linewidth=4,label = L"\eta_\tau^2=0.1",color = "green")

 plot(fig3,fig1,size = [600,500],
 layout = grid(2, 1, heights=[0.9 ,0.1]),
  labelfontsize=14,)

Plots.savefig("figure0/optimal_noise_timeseries.png")


##
using Interpolations
tt = collect(range(0,10000,200000))
sol = sol1
nodes = (sol.t,)
expr0 = sol[4,:]
itp_fun = Interpolations.interpolate(nodes,expr0, Gridded(Constant{Previous}()))
mRNA_expr= itp_fun(tt)

using StatsBase
lag = 100
cor1 = StatsBase.autocor(mRNA_expr,1:lag)


sol = sol2
nodes = (sol.t,)
expr0 = sol[4,:]
itp_fun = Interpolations.interpolate(nodes,expr0, Gridded(Constant{Previous}()))
mRNA_expr= itp_fun(tt)
cor2 = StatsBase.autocor(mRNA_expr,1:lag)


sol = sol3
nodes = (sol.t,)
expr0 = sol[4,:]
itp_fun = Interpolations.interpolate(nodes,expr0, Gridded(Constant{Previous}()))
mRNA_expr= itp_fun(tt)
cor3 = StatsBase.autocor(mRNA_expr,1:lag)

plot(1:lag,cor1,linewidth=5,label = L"\eta_\tau^2=10",
grid=0,framestyle=:box,palette = :tab10,
 labelfontsize=14,legendfontsize=14,tickfontsize=12,dpi=600,
  ylabel = "Autocorrelation",xlabel = "Lag",size = (500,400))
plot!(1:lag,cor2,linewidth = 5,label = L"\eta_\tau^2=1",)
plot!(1:lag,cor3,linewidth = 5,label = L"\eta_\tau^2=0.1")

Plots.savefig("figure0/optimal_noise_autocorrelation.png")
