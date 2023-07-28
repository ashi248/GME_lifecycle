using LinearAlgebra
using BAT, DensityInterface, IntervalSets
using ValueShapes
using JLD2
using Plots
using Interpolations
using Random, Distributions
using StatsBase
include("generating_data.jl")
include("inference_function.jl")


## generate data
function generating_data(N)
      kon = 1; koff = 0.5;alpha = 10; r = 1; rou1 = 20;  mu = 1
      nasRNA = zeros(N)
      for i in 1:N
            jsol = generate_data_mature(kon,koff,rou1,r,alpha,mu)
            tt = collect(range(0,100,101))
            nodes = (jsol.t,)
            mRNA = jsol[3,:]
            mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
            mRNA_expr = mRNA_itp(tt)
            nasRNA[i]  = mRNA_expr[100]
      end
     return(nasRNA)
end
data0 = generating_data(3000)

function inference_bayes(data0,N)
      # bayesian inference
      data = data0[1:N]
      likelihood = let d = data, f = fit_function
            logfuncdensity(function(params)
                  NT = Int(maximum(d))
                  count_data = abs.(f(params, NT)[:,1])
                  ind = @. Int(d+1)
                  n_data=length(d)
                  likeli = sum(log.(count_data[ind]))
                  return likeli
            end)
      end

      prior = NamedTupleDist(
            α = Uniform(log(0.1), log(20)),
            λ1 = Uniform(log(1), log(50)),
            k0 = Uniform(log(0.1), log(10)),
            k1 = Uniform(log(0.1), log(10)),
      )

      parshapes = varshape(prior)
      posterior = PosteriorMeasure(likelihood, prior)

      samples =bat_sample(posterior,MCMCSampling(mcalg = MetropolisHastings(),
      nsteps = 10^5)).result
      return(samples)
end

sample1000 = inference_bayes(data0,1000)
sample2000 = inference_bayes(data0,2000)
sample3000 = inference_bayes(data0,3000)


plot(
      sample2000,
      mean = false,
      std = false,
      globalmode = true,
      marginalmode = false,
      nbins = 40,
)


save_object("data/sample1000.jld2",sample1000)
save_object("data/sample2000.jld2",sample2000)
save_object("data/sample3000.jld2",sample3000)

##
# plot
sample1000 = load_object("data/sample1000.jld2")
sample2000 = load_object("data/sample2000.jld2")
sample3000 = load_object("data/sample3000.jld2")


flat_v = BAT.flatview(unshaped.(sample1000).v)
flat_logd = BAT.flatview(unshaped.(sample1000).logd)
result_1000 = convert(Matrix, hcat(transpose(flat_v), flat_logd))


flat_v = BAT.flatview(unshaped.(sample2000).v)
flat_logd = BAT.flatview(unshaped.(sample2000).logd)
result_2000 = convert(Matrix, hcat(transpose(flat_v), flat_logd))

flat_v = BAT.flatview(unshaped.(sample3000).v)
flat_logd = BAT.flatview(unshaped.(sample3000).logd)
result_3000 = convert(Matrix, hcat(transpose(flat_v), flat_logd))



using StatsPlots
density(result_1000[:, 1], lw = 4,label = "M=1000")
density!(result_2000[:, 1], lw = 4,label = "M=2000")
density!(result_3000[:, 1], lw = 4,label = "M=3000")
vline!([log(10)], lw = 5,color=:black,label = "True value")
plot!(xlabel = "Log(α)", ylabel = "Posterior density", grid=0,framestyle=:box,
dpi=600,legendfontsize = 12,tickfontsize = 12,xlims = [0,3.5],
labelfontsize=14,size = (500,400),legend = :topleft)


Plots.savefig("figure0/post_distribution_alpha.png")

#
density(result_1000[:, 2], lw = 4,legend = :topleft,label = "M=1000")
density!(result_2000[:, 2], lw = 4,legend = :topleft,label = "M=2000")
density!(result_3000[:, 2], lw = 4,legend = :topleft,label = "M=3000")
vline!([log(20)], lw = 5,color=:black,label = "True value")
plot!(xlabel = "Log(λ1)", ylabel = "Posterior density", grid=0,framestyle=:box,
dpi=600,legendfontsize = 12,tickfontsize = 12,xlims = [2.92,3.05],
labelfontsize=14,size = (500,400),legend = :topleft)

Plots.savefig("figure0/post_distribution_lambda.png")


density(result_1000[:, 3], lw = 4,legend = :topleft,label = "M=1000")
density!(result_2000[:, 3], lw = 4,legend = :topleft,label = "M=2000")
density!(result_3000[:, 3], lw = 4,legend = :topleft,label = "M=3000")

vline!([log(1)], lw = 5,color=:black,label = "True value")

plot!(xlabel = "Log(k0)", ylabel = "Posterior density", grid=0,framestyle=:box,
xlims = [-0.4,0.4],ylims = [0,6],dpi=600,legendfontsize = 12,tickfontsize = 12,
labelfontsize=14,size = (500,400),legend = :topleft)

Plots.savefig("figure0/post_distribution_k0.png")


density(result_1000[:, 4], lw = 4,legend = :topleft,label = "M=1000")
density!(result_2000[:, 4], lw = 4,legend = :topleft,label = "M=2000")
density!(result_3000[:, 4], lw = 4,legend = :topleft,label = "M=3000")
vline!([log(0.5)], lw = 5,color=:black,label = "True value")

plot!(xlabel = "Log(k1)", ylabel = "Posterior density", grid=0,framestyle=:box,
xlims = [-1.5,-0.1],ylims = [0,4.5],dpi=600,legendfontsize = 12,tickfontsize = 12,
labelfontsize=14,size = (500,400),legend = :topleft)

Plots.savefig("figure0/post_distribution_k1.png")
