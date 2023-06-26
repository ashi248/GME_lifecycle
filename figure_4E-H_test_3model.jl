using DelaySSAToolkit
using Random, Distributions
using Catalyst, Plots
using StatsBase
using Interpolations
using DataFrames
using JLD2

include("inference_function.jl")
include("generating_data.jl")


##
# generating test data
M0 = 300
Kon = rand(LogUniform(0.1,2),M0)
fon = rand(Uniform(0.2,0.8),M0)
Koff = Kon .* (1 .-fon)./fon

rou = rand(LogUniform(10,50),M0)
alpha0 = rand(LogUniform(0.1,10),M0)
Parameter = DataFrame(kon = Kon,koff = Koff,rou = rou,alpha = alpha0,mu = 1)

function generating_data(N=5000,kon=1,koff=0.5,rou=20,alpha=10,mu=1)
      kon; koff;alpha; r = 1; rou; mu
      nasRNA = zeros(N)
      for i in 1:N
            jsol = generate_data_mature(kon,koff,rou,r,alpha,mu)
            tt = collect(range(0,100,101))
            nodes = (jsol.t,)
            mRNA = jsol[3,:]
            mRNA_itp = Interpolations.interpolate(nodes,mRNA, Gridded(Constant{Previous}()))
            mRNA_expr = mRNA_itp(tt)
            nasRNA[i]  = mRNA_expr[100]
      end
     return(nasRNA)
end


N = 5000
mRNA_nas = zeros(N,M0)

for j in 1:M0
     print(j)
        nas = generating_data(N,Kon[j],Koff[j],rou[j],alpha0[j],1)
        mRNA_nas[:,j] = nas
end

save_object("data/test_data_3model.jld2",(mRNA_nas,Parameter))


##
using Random, Distributions
using Plots
using StatsBase
using Interpolations
using DataFrames
using JLD2

include("inference_function.jl")
mRNA_nas,Parameter = load_object("data/test_data_3model.jld2")

parameter_preds = zeros(M0,4)
using BlackBoxOptim

for k in 1:M0
    print(k)
    nas = mRNA_nas[:,k]
    nas_freq= sort(countmap(nas))
    cost_function(p) = Log_Likelihood_nascent(nas_freq, p)
    bound1 = Tuple{Float64, Float64}[(log(0.1), log(10)),(log(10), log(50)),
    (log(0.01), log(10)),(log(0.01), log(10))]
    result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 1e4)
    results =  result.archive_output.best_candidate
    parameter_preds[k,:] = exp.(results)
end

save_object("data/infer_result_nonMarkov.jld2",parameter_preds)

##

include("inference_function.jl")
parameter_preds_exp = zeros(M0,3)
using BlackBoxOptim

for k in 1:M0
    print(k)
    nas = mRNA_nas[:,k]
    nas_freq= sort(countmap(nas))
    cost_function(p) = Log_Likelihood_nascent_markov(nas_freq, p)

    bound1 = Tuple{Float64, Float64}[(log(10), log(50)),
    (log(0.01), log(10)),(log(0.01), log(10))]
    result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 1e4)
    results =  result.archive_output.best_candidate
    parameter_preds_exp[k,:] = exp.(results)
end

save_object("data/infer_result_Markov.jld2",parameter_preds_exp)

##
# deterministic delay
include("inference_function.jl")
parameter_preds_fix = zeros(M0,3)
using BlackBoxOptim

for k in 1:M0
    print(k)
    nas = mRNA_nas[:,k]
    nas_freq= sort(countmap(nas))
    cost_function(p) = Log_Likelihood_nascent_fix(nas_freq, p)

    bound1 = Tuple{Float64, Float64}[(log(10), log(50)),
    (log(0.01), log(10)),(log(0.01), log(10))]
    result = bboptimize(cost_function;SearchRange = bound1,MaxSteps = 1e4)
    results =  result.archive_output.best_candidate
    parameter_preds_fix[k,:] = exp.(results)
end

save_object("data/infer_result_fix.jld2",parameter_preds_fix)

##

mRNA_nas,parms_true = load_object("data/test_data_3model.jld2")
result0= load_object("data/infer_result_nonMarkov.jld2")
result1 = result0[:,2:4]
result2 = load_object("data/infer_result_Markov.jld2")
result3 = load_object("data/infer_result_fix.jld2")

function MRE_calulate1(result)
    MRE= zeros(M0,4)
    for i in 1:M0
        pest = result[i,:]
        ptrue = parms_true[i,:]
        mr1 = abs(pest[1]-ptrue.rou)/ptrue.rou
        mr2 = abs(pest[2]-ptrue.kon)/ptrue.kon
        mr3 = abs(pest[3]-ptrue.koff)/ptrue.koff
        mr4 = (mr1+mr2+mr3)/3
        MRE[i,:] = [mr1,mr2,mr3,mr4]
    end
    return(MRE)
end

MRE1  = MRE_calulate1(result1)
MRE2  = MRE_calulate1(result2)
MRE3  = MRE_calulate1(result3)
mean(MRE1[:,1])
mean(MRE2[:,1])
mean(MRE3[:,1])

##
# box plot
using StatsPlots
mre = [MRE1[:,1];MRE2[:,1];MRE3[:,1]]
boxplot(repeat(["Model I","Model II","Model III"],inner=300),mre,
label = "",color = :yellow,linewidth=2)
plot!(ylabel = "Relative error", grid=0,framestyle=:box,dpi=600,tickfontsize = 14,
labelfontsize=14,size = (500,400),yscale = :log)
Plots.savefig("figure0/model_comparison_rou.png")

using StatsPlots
mre = [MRE1[:,2];MRE2[:,2];MRE3[:,2]]
boxplot(repeat(["Model I","Model II","Model III"],inner=300),mre,
label = "",color = :yellow,linewidth=2)
plot!(ylabel = "Relative error", grid=0,framestyle=:box,dpi=600,tickfontsize = 14,
labelfontsize=14,size = (500,400),yscale = :log)
Plots.savefig("figure0/model_comparison_kon.png")

using StatsPlots
mre = [MRE1[:,3];MRE2[:,3];MRE3[:,3]]
boxplot(repeat(["Model I","Model II","Model III"],inner=300),mre,
label = "",color = :yellow,linewidth=2)
plot!(ylabel = "Relative error", grid=0,framestyle=:box,dpi=600,tickfontsize = 14,
labelfontsize=14,size = (500,400),yscale = :log)
Plots.savefig("figure0/model_comparison_koff.png")


using StatsPlots
mre = [MRE1[:,4];MRE2[:,4];MRE3[:,4]]
boxplot(repeat(["Model I","Model II","Model III"],inner=300),mre,
label = "",color = :yellow,linewidth=2)
plot!(ylabel = "Mean relative error", grid=0,framestyle=:box,dpi=600,tickfontsize = 14,
labelfontsize=14,size = (500,400),yscale = :log)
Plots.savefig("figure0/model_comparison_MRE.png")

##
