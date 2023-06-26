using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots
include("delay_function.jl")
include("FSP_distribution.jl")


#α = 1; t0 = 1; k0 = 1; k1 = 1;  λ= 10
# gamma delay
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
      saveat = collect(range(0,30,31))
      #saveat = [0 400]
      H = delay_mature_gamma
      model(du, u, p, t) = model_4state(du, u, p, t, H)
      prob1 = FSP_distr_4state(model, p, tspan, saveat)
      return (prob1)
end


prob0 = gamma_distribution_4state(4, 4, 0.5, 1, 20)
N = size(prob0)[1] - 1
bins = collect(0:N);

#Plots.plot(bins, prob0[:,1], linewidth = 5, label = "L=30,T=0.3",xlims = [0, 50])
Plots.plot(bins, prob0[:,30], linewidth = 5,
label = "L=30,T=0.3",xlims = [0, 50])

##
using Catalyst, Plots, DiffEqJump
using StatsBase
using Distributions

rs = @reaction_network begin
  c11, G00 --> G01
  c12, G01 --> G00
  c13, G01 --> G11
  c14, G11 --> G01
  c15, G11 --> G10
  c16, G10 --> G11
  c17, G00 --> G10
  c18, G10 --> G00

  c21, G00 --> G00 + m1
  c22, G01 --> G01 + m1
  c23, G11 --> G11 + m1
  c24, G10 --> G10 + m1

  c31, m1 --> m2
  c32, m2 --> m3
  c33, m3 --> m4
  c34, m4 --> m
  c4, m --> 0
end c11 c12 c13 c14 c15 c16 c17 c18 c21 c22 c23 c24 c31 c32 c33 c34 c4

k0 = 0.5; k1 = 1; r = 20; d1 = 1; d2=1;
p = (:c11 => k0, :c12 => k1, :c13 => k0, :c14 =>k1, :c15 =>k1,:c16 =>k0, :c17 =>k0,:c18 =>k1,
:c21 => 0,:c22 => r,:c23 => 2*r,:c24 => r,:c31 => d1,:c32 => d1,:c33 => d1,:c34 => d1,:c4 => d2)
#tspan = (0., 200000.)
tspan = (0., 20000.)

function geneSSA(x)
  state = zeros(4)
  state[x] = 1
  u0    = [:G00 => state[1], :G01 => state[2], :G10 => state[3], :G11 => state[4], :m1 => 0,:m2 => 0,:m3 => 0,:m4 => 0,:m => 0]
  # solve JumpProblem
  dprob = DiscreteProblem(rs, u0, tspan, p)
  jprob = JumpProblem(rs, dprob, Direct())
  jsol = solve(jprob, SSAStepper())
  return(jsol)
end

jsol = geneSSA(1)

using Interpolations
tt = collect(range(0,20000,20001))
nodes = (jsol.t,)
mRNA = jsol[9,:]
mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
mRNA_expr = mRNA_itp(tt)
mRNA_prob = proportionmap(mRNA_expr)
plot!(mRNA_prob,shape = [:circle],markersize=8)


##

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
      saveat = collect(range(0,30,31))
      #saveat = [0 400]
      H = delay_mature_gamma
      model(du, u, p, t) = model_4state(du, u, p, t, H)
      prob1 = FSP_distr_4state(model, p, tspan, saveat)
      return (prob1)
end

prob0 = gamma_distribution_4state(4, 4, 0.5, 0.5, 20)
N = size(prob0)[1] - 1
bins = collect(0:N);
Plots.plot(bins, prob0[:,5], linewidth = 5, label = "Theory",xlims = [0, 30])
#

k0 = 0.5; k1 = 0.5; r = 20; d1 = 1; d2=1;
p = (:c11 => k0, :c12 => k1, :c13 => k0, :c14 =>k1, :c15 =>k1,:c16 =>k0, :c17 =>k0,:c18 =>k1,
:c21 => 0,:c22 => r,:c23 => 2*r,:c24 => r,:c31 => d1,:c32 => d1,:c33 => d1,:c34 => d1,:c4 => d2)
tspan = (0., 30.)

function geneSSA(x)
  state = zeros(4)
  state[x] = 1
  u0    = [:G00 => state[1], :G01 => state[2], :G10 => state[3], :G11 => state[4], :m1 => 0,:m2 => 0,:m3 => 0,:m4 => 0,:m => 0]
  # solve JumpProblem
  dprob = DiscreteProblem(rs, u0, tspan, p)
  jprob = JumpProblem(rs, dprob, Direct())
  jsol = solve(jprob, SSAStepper())
  return(jsol)
end

using Interpolations
tt = collect(range(0,30,31))
N = 400000
mRNA_expr = zeros(N,length(tt))

W = [-2*k0   k1      0    k1;
       k0  -(k0+k1)  k1    0;
       0      k0     -2*k1  k0;
        1     1        1    1]
w0 = [0;0;0;1]
p0 = W\w0

sample1 = rand(Categorical(p0),N)
sample2 = repeat([1 2 3 4],1,100000)


for i in 1:N
  print(i)
  x = sample2[i]
  jsol = geneSSA(x)
  nodes = (jsol.t,)
  mRNA = jsol[9,:]
  mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
  mRNA_expr[i,:] = mRNA_itp(tt)
end


#Plots.plot(bins, prob0[:,1], linewidth = 5, label = "L=30,T=0.3",xlims = [0, 50])
Plots.plot(bins, prob0[:,3],linewidth=5,label = "model",xlims = [0,10],
xlabel="Cytoplasmic mRNA#", ylabel="Probability",
labelfontsize=14,dpi=600,legendfontsize = 12,tickfontsize = 12,
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box,size = (500,400))

mRNA_prob = proportionmap(mRNA_expr[:,3])
scatter!(mRNA_prob,shape = [:circle],markersize=8,label="SSA")
Plots.savefig("figure0/4state_t3.png")

#Plots.plot(bins, prob0[:,1], linewidth = 5, label = "L=30,T=0.3",xlims = [0, 50])
Plots.plot(bins, prob0[:,5],linewidth=5,label = "model",xlims = [0,30],
xlabel="Cytoplasmic mRNA#", ylabel="Probability",
labelfontsize=14,dpi=600,legendfontsize = 12,tickfontsize = 12,
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box,size = (500,400))

mRNA_prob = proportionmap(mRNA_expr[:,5])
scatter!(mRNA_prob,shape = [:circle],markersize=8,label="SSA")
Plots.savefig("figure0/4state_t5.png")

#Plots.plot(bins, prob0[:,1], linewidth = 5, label = "L=30,T=0.3",xlims = [0, 50])
Plots.plot(bins, prob0[:,10],linewidth=5,label = "model",xlims = [0,50],
xlabel="Cytoplasmic mRNA#", ylabel="Probability",
labelfontsize=14,dpi=600,legendfontsize = 12,tickfontsize = 12,
grid=0,fillrange = 0, fillalpha = 0.25,framestyle=:box,size = (500,400))

mRNA_prob = proportionmap(mRNA_expr[:,10])
scatter!(mRNA_prob,shape = [:circle],markersize=8,label="SSA")

Plots.savefig("figure0/4state_t10.png")
