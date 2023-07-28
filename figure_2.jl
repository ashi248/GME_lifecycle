using Distributions, Distances
using DelimitedFiles,Plots
using QuadGK
include("delay_function.jl")
include("FSP_distribution.jl")
include("noise.jl")


##
# figure 2A
# noise as function of noise and average retention time
α1 = 1; θ1 = 1/α1;
p1  = [40,0,0.5,0.5]     #[λ1,λ2,k0,k1]
p2 = [α1,θ1]  # parameter for delay function
H = delay_nascent_gamma
Mean,Var = gamma_distribution_noise(H,p1,p2)
noise0 = Var/Mean^2



# heatmap of noise
noise= exp.(collect(range(log(0.1),log(20),100)))
LL = 1 ./ noise
tt = exp.(collect(range(log(0.1),log(10),100)))
mRNA_noise = Matrix(undef,length(LL),length(tt))
mRNA_mean = Matrix(undef,length(LL),length(tt))
ratio = Matrix(undef,length(LL),length(tt))

for i in 1:length(LL)
      print(i)
      for j in 1:length(tt)
          α1 = 1*LL[i]; θ1 = tt[j]/α1; r=1;
          α2 = 1; θ2 = 1/α2;
          p1  = [40,0,0.5,0.5]    #[λ1,λ2,k0,k1]
          p2 = [α1,θ1,r]
          H = delay_mature_gamma
          Mean,Var = gamma_distribution_noise(H,p1,p2)
          noise1 = Var/Mean^2
          mRNA_noise[i,j] = noise1
          mRNA_mean[i,j] = Mean
          ratio[i,j] = noise1/noise0
      end
end



noise= exp.(collect(range(log(0.1),log(20),100)))
LL = 1 ./ noise
tt = exp.(collect(range(log(0.1),log(10),100)))
using Plots
pyplot()


Plots.heatmap(noise,tt, ratio',c =:viridis,xlabel = "Noise of elongation",
ylabel = "Average of elongation",yscale =:log, xscale =:log,framestyle=:box,
xlims = [0.1,20],ylims = [0.1,10],labelfontsize=14,dpi=600,
tickfontsize = 12,legendfontsize = 14,
colorbar_title = "Ratio of mature RNA noise",colorbar_titlefontsize = 14,
colorbar_tickfontsize=12)


##
# figure 2A identify the shape of distribution
include("find_peak.jl")
noise= exp.(collect(range(log(0.1),log(20),100)))
LL = 1 ./ noise
tt = exp.(collect(range(log(0.1),log(10),100)))
type = Matrix(undef,length(LL),length(tt))
for i in 1:length(LL)
      print(i)
      for j in 1:length(tt)
          α = LL[i]; θ = tt[j]/α; r=1
          p1  = [40,0,0.5,0.5]    #[λ1,λ2,k0,k1]
          p2 = [α,θ,r]
          p3 = 50
          p = [p1,p2,p3]
          tspan = (0.0,100.0);
          saveat = [100]
          H = delay_mature_gamma;
          model(du,u,p,t) = model_2state(du,u,p,t,H)
          probs = FSP_distr(model,p,tspan,saveat)
          type[i,j] = find_peak(probs)
      end
end


type_t = type'
yy = zeros(100)
for i in 1:100
      a =  type_t[:,i]
      idx = a.!=3
      ps = indexin(1,idx)
      yy[i] = tt[ps][1]
end

xx2 = []
yy2 = []
for i in 1:100
      a =  type_t[:,i]
      df1 = diff(a)
      ps = indexin(-2,df1)[1]!= nothing
      if (ps)
        d1 = i
        d2 = indexin(-2,df1)[1]
        xx2 = append!(xx2,d1)
        yy2 = append!(yy2,d2)
      end
end

xx3 = [xx2[4:length(xx2)];79]
yy3 = [yy2[4:length(xx2)];99]


using Loess
xs = noise
ys = yy
model1 = loess(xs, ys, span=0.2)
us = range(extrema(xs)...; step = 0.01)
vs = Loess.predict(model1, us)
plot!(us, vs,color="white",lw=4,legend=:none)


xs1 = noise[xx3]
ys1 = tt[yy3.+1]
model2 = loess(xs1, ys1, span=0.4)
us1 = range(extrema(xs1)...; step = 0.01)
vs1 = Loess.predict(model2, us1)
plot!(us1, vs1,color="white",lw=4,legend=:none)


us2 = collect(range(0.1,20,step=0.01))
vs2 = 1 ./us2
vline!([1],color="red",lw=4,legend=:none,ls=:dash)


using Plots
Plots.savefig("figure0/heat_noise_mature.png")


##
#Figure 2
# (B)(C)(D)
include("FSP_distribution.jl")

##
function mature_mRNA_distribution(H,L=1,t0=1,r=1,k0=0.5,k1=0.5,λ1 = 40,λ2 = 0,time = 800)
      #L=1;t0=1;k0=0.5;k1=0.5;λ1 = 40;λ2 = 0;
      α = L; θ = t0/L; r = r;
      N = 100;
      p1  = [λ1,λ2,k0,k1]
      p2 = [α,θ,r]
      p3 = N
      p = [p1,p2,p3]

      tspan = (0.0,800.0);
      saveat = [time]
      model(du,u,p,t) = model_2state(du,u,p,t,H)
      probs = FSP_distr(model,p,tspan,saveat)
      return(probs)
end

# figure 2D
function mature_plot(L = 0.1,k0=0.5,k1=0.5)
      t0=2;r=1;λ1 = 40;λ2 = 0;
      H = delay_mature_gamma;
      prob = mature_mRNA_distribution(H,L,t0,r,k0,k1,λ1,λ2)
      N = length(prob)-1
      bins = collect(0:N)
      fig = plot(bins, prob,size = (500,400),
      xlims = [0,60],dpi=600, legend = :none,
      grid=0,fillrange = 0, fillalpha = 0.25,
      framestyle=:box)
      return(fig)
end

p1 = mature_plot(10, 0.5, 0.2)
p2 = mature_plot(1, 0.5, 0.2)
p3 = mature_plot(0.1, 0.5, 0.2)

p4 = mature_plot(10, 0.5, 0.5)
p5 = mature_plot(1, 0.5, 0.5)
p6 = mature_plot(0.1, 0.5, 0.5)

p7 = mature_plot(10, 0.5, 0.8)
p8 = mature_plot(1, 0.5, 0.8)
p9 = mature_plot(0.1, 0.5, 0.8)

plot(p1,p2,p3,p4,p5,p6,p7,p8,p9,layout=9,linewidth=2)

Plots.savefig("figure0/figure_S1.png")

##

function lognormal_distribution(μ=2,σ=1)
      α = 1; θ = 2/α;
      k0=0.2;k1=0.2;λ1 = 40;λ2 = 0
      N = 100;
      p1  = [λ1,λ2,k0,k1]
      p2 = [μ,σ,α,θ]
      p3 = N
      p = [p1,p2,p3]
      H = delay_mature_total_log
      tspan = (0.0,400.0)
      saveat = [400]
      model(du,u,p,t) = model_2state(du,u,p,t,H)
      probs = FSP_distr(model,p,tspan,saveat)
      return(probs)
end


prob = lognormal_distribution(2,0.1)
N = length(prob)-1
bins = collect(0:N);
plot(bins, prob,linewidth=4,label = "σ=0.1", size = (500,400),
xlims = [0,100],dpi=600,
grid=0,framestyle=:box,labelfontsize=14,tickfontsize = 12,legendfontsize = 14,)



prob = lognormal_distribution(2,1)
N = length(prob)-1
bins = collect(0:N);
plot!(bins, prob,linewidth=4,label = "σ=1", size = (500,400),
xlims = [0,100],dpi=600,
grid=0,framestyle=:box,labelfontsize=14,tickfontsize = 12,)



prob = lognormal_distribution(2,10)
N = length(prob)-1
bins = collect(0:N);
plot!(bins, prob,linewidth=4,label = "σ=10", size = (500,400),
xlims = [0,100],dpi=600,
grid=0,framestyle=:box,labelfontsize=14,tickfontsize = 12,)
xlabel!("Mature RNA#"); ylabel!("Probability")

Plots.savefig("figure0/figure_S2.png")
