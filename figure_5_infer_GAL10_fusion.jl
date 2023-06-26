using DelimitedFiles
total_set = 8
histo_data_vec = Array{Any,1}(undef,total_set)
histo_data_vec[1] = vec(readdlm("data/Signal_data/Dataset#1/20170511_YTL047A1_GAL_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G1cells.txt", ',', Float64))
histo_data_vec[2] = vec(readdlm("data/Signal_data/Dataset#2/20170511_YTL047A2_GAL_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G1cells.txt", ',', Float64))
histo_data_vec[3] = vec(readdlm("data/Signal_data/Dataset#3/20170824_YTL047A1_gal_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G1cells.txt", ',', Float64))
histo_data_vec[4] = vec(readdlm("data/Signal_data/Dataset#4/20170824_YTL047A2_gal_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G1cells.txt", ',', Float64))

histo_data_vec[5] = vec(readdlm("data/Signal_data/Dataset#1/20170511_YTL047A1_GAL_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G2cells.txt", ',', Float64))
histo_data_vec[6] = vec(readdlm("data/Signal_data/Dataset#2/20170511_YTL047A2_GAL_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G2cells.txt", ',', Float64))
histo_data_vec[7] = vec(readdlm("data/Signal_data/Dataset#3/20170824_YTL047A1_gal_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G2cells.txt", ',', Float64))
histo_data_vec[8] = vec(readdlm("data/Signal_data/Dataset#4/20170824_YTL047A2_gal_PP7-cy3_1_intensity_brightest_txpnsite_cy3_G2cells.txt", ',', Float64))

histo_data_vec_merged = Array{Any,1}(undef,4)
histo_data_vec_merged[1] = readdlm("data/DAPI_content/20170511_YTL047A1_GAL_PP7-cy3_1_dapiContent_nascentTSintensity_allcells.txt")[:,2]
histo_data_vec_merged[2] = readdlm("data/DAPI_content/20170511_YTL047A2_GAL_PP7-cy3_1_dapiContent_nascentTSintensity_allcells.txt")[:,2]
histo_data_vec_merged[3] = readdlm("data/DAPI_content/20170824_YTL047A1_gal_PP7-cy3_1_dapiContent_nascentTSintensity_allcells.txt")[:,2]
histo_data_vec_merged[4] = readdlm("data/DAPI_content/20170824_YTL047A2_gal_PP7-cy3_1_dapiContent_nascentTSintensity_allcells.txt")[:,2]


normalizing_valuev1 = vec(readdlm("data/Signal_data/cyto_normalization_values.txt",Float64))[1:4]
normalizing_value = repeat(normalizing_valuev1,2);
histo_data_vec_normalized = histo_data_vec./normalizing_value;



for i in 1:4
    filter!(x->x.!="N/A",histo_data_vec_merged[i])
end

normalizing_valuev1 = vec(readdlm("data/Signal_data/cyto_normalization_values.txt",Float64))[1:4];
normalizing_value = repeat(normalizing_valuev1,1);
histo_data_vec_merged_normalized = histo_data_vec_merged./normalizing_value;




##
using  LinearAlgebra
using BAT, DensityInterface, IntervalSets
using ValueShapes
include("inference_function_GAL10.jl")

function GAL10_inference(data)
      data = fusion(data)
      NT = Int(ceil(maximum(data)) + 1)
      filter = hcat(convolve_uniform([862, 2200], NT+2; Δ=0.01)...)

      likelihood = let d = data, f = fit_function, filter = filter
            ## Histogram counts for each bin as an array:
            logfuncdensity(function (params)
                  NT = Int(ceil(maximum(d)) + 1)
                  count_data = f(params, NT)[:,1]
                  estimate_signal  = convolve_filter(filter, count_data)
                  n_data=length(d)
                  estimate_signal_1 = @. max(estimate_signal, 0)
                  estimate_signal_fusion = signal_fusion(estimate_signal_1)
                  ind = @. Int(floor(d) + 1) # because the index is from 0
                  likeli = sum(log.(estimate_signal_fusion[ind] .+ 0.1 / n_data))
                  return likeli
            end)
      end



      prior = NamedTupleDist(
            α = Uniform(log(0.1), log(40)),
            λ1 = Uniform(log(0.1), log(50)),
            k0 = Uniform(log(0.1), log(10)),
            k1 = Uniform(log(0.1), log(10)),
      )


      parshapes = varshape(prior)
      posterior = PosteriorMeasure(likelihood, prior)

      samples =bat_sample(posterior,MCMCSampling(mcalg = MetropolisHastings(),
      nsteps = 20000)).result


      samples_mode = mode(samples)
      findmode_result = bat_findmode(
            posterior,
            MaxDensityNelderMead(init = ExplicitInit([samples_mode])),
      )
      fit_par = findmode_result.result
      params = [exp(fit_par.α),exp(fit_par.λ1),exp(fit_par.k0),exp(fit_par.k1)]
      return([samples,params])
end

data  = copy(histo_data_vec_normalized[1])
samples1, params1 = GAL10_inference(data)


data  = copy(histo_data_vec_normalized[2])
samples2, params2 = GAL10_inference(data)

data  = copy(histo_data_vec_normalized[3])
samples3, params3 = GAL10_inference(data)

data  = copy(histo_data_vec_normalized[4])
samples4, params4 = GAL10_inference(data)


using Plots
plot(
      samples1,
      mean = false,
      std = false,
      globalmode = true,
      marginalmode = false,
      nbins = 50,
      size = (900,600),
      labelfontsize=12,
      dpi=600,
      vsel_label = ["log α","log λ1","log k0","log k1"]
)

Plots.savefig("figure0/GAL10_post_distribution_data1.png")

##
# comparing data and estimation
include("inference_function_GAL10.jl")

function compare_plot(data,param)
      data = fusion(data)
      histogram(data,bins=0:70,normalize=:probability,
      color = "skyblue",labels="Data",labelfontsize=14,dpi=600,legendfontsize = 10,
      tickfontsize = 10,grid=0,size = (500,400))

      NT = Int(ceil(maximum(data)) + 1)
      filter1 = hcat(convolve_uniform([862, 2200], NT+2; Δ=0.01)...)
      param_log = log.(param)
      param = (α=param_log[1], λ1=param_log[2], k0=param_log[3], k1=param_log[4])
      count_data = fit_function(param, 50)[:,1]
      estimate_signal  = convolve_filter(filter1, count_data)
      estimate_signal_fusion = signal_fusion(estimate_signal)

      L0 = length(estimate_signal_fusion)
      Plots.plot!(collect((4-0.5):(L0-0.5)), estimate_signal_fusion[4:L0], linewidth = 5,label = "Model",
      xlabel = "Signal intensity of nascent RNA",ylabel = "frequency",
      color = :red)
end

data  = histo_data_vec_normalized[1]
param = params1
compare_plot(data,param)
Plots.savefig("figure0/GAL10_histogram_data1.png")

data  = histo_data_vec_normalized[2]
param = params2
compare_plot(data,param)
Plots.savefig("figure0/GAL10_histogram_data2.png")

data  = histo_data_vec_normalized[3]
param = params3
compare_plot(data,param)
Plots.savefig("figure0/GAL10_histogram_data3.png")

data  = histo_data_vec_normalized[4]
param = params4
compare_plot(data,param)
Plots.savefig("figure0/GAL10_histogram_data4.png")


##
using  LinearAlgebra
using BAT, DensityInterface, IntervalSets
using ValueShapes
include("inference_function_GAL10.jl")

function GAL10_inference_G2(data)
      data = fusion(data)
      NT = Int(ceil(maximum(data)) + 1)
      filter = hcat(convolve_uniform([862, 2200], NT+2; Δ=0.01)...)

      likelihood = let d = data, f = fit_function, filter = filter
            ## Histogram counts for each bin as an array:
            logfuncdensity(function (params)
                  NT = Int(ceil(maximum(d)) + 1)
                  count_data = f(params, NT)[:,1]
                  estimate_signal  = convolve_filter(filter, count_data)
                  estimate_signal_2 = DSP.conv(estimate_signal,estimate_signal)
                  n_data=length(d)
                  estimate_signal_1 = @. max(estimate_signal_2, 0)
                  estimate_signal_fusion = signal_fusion(estimate_signal_1)
                  ind = @. Int(floor(d) + 1) # because the index is from 0
                  likeli = sum(log.(estimate_signal_fusion[ind] .+ 0.1 / n_data))
                  return likeli
            end)
      end



      prior = NamedTupleDist(
            α = Uniform(log(0.1), log(40)),
            λ1 = Uniform(log(0.1), log(50)),
            k0 = Uniform(log(0.1), log(10)),
            k1 = Uniform(log(0.1), log(10)),
      )


      parshapes = varshape(prior)
      posterior = PosteriorMeasure(likelihood, prior)

      samples =bat_sample(posterior,MCMCSampling(mcalg = MetropolisHastings(),
      nsteps = 50000)).result


      samples_mode = mode(samples)
      findmode_result = bat_findmode(
            posterior,
            MaxDensityNelderMead(init = ExplicitInit([samples_mode])),
      )
      fit_par = findmode_result.result
      params = [exp(fit_par.α),exp(fit_par.λ1),exp(fit_par.k0),exp(fit_par.k1)]
      return([samples,params])
end

data  = copy(histo_data_vec_normalized[5])
samples5, params5 = GAL10_inference_G2(data)

data  = copy(histo_data_vec_normalized[6])
samples6, params6 = GAL10_inference_G2(data)

data  = copy(histo_data_vec_normalized[7])
samples7, params7 = GAL10_inference_G2(data)

data  = copy(histo_data_vec_normalized[8])
samples8, params8 = GAL10_inference_G2(data)


using Plots
plot(
      samples5,
      mean = false,
      std = false,
      globalmode = true,
      marginalmode = false,
      nbins = 50,
      size = (900,600),
      labelfontsize=12,
      dpi=600,
      vsel_label = ["log α","log λ1","log k0","log k1"]
)

Plots.savefig("figure0/GAL10_post_distribution_data5.png")



include("inference_function_GAL10.jl")

function compare_plot_G2(data,param)
      data = fusion(data)
      NT = Int(ceil(maximum(data)) + 1)
      filter1 = hcat(convolve_uniform([862, 2200], NT+2; Δ=0.01)...)

      histogram(data,bins=0:70,normalize=:probability,
      color = "skyblue",labels="Data",labelfontsize=14,dpi=600,legendfontsize = 10,
      tickfontsize = 10,grid=0,size = (500,400))

      param_log = log.(param)
      param0 = (α=param_log[1], λ1=param_log[2], k0=param_log[3], k1=param_log[4])
      count_data = fit_function(param0, NT)[:,1]
      estimate_signal  = convolve_filter(filter1, count_data)
      estimate_signal_2 = DSP.conv(estimate_signal,estimate_signal)
      estimate_signal_1 = @. max(estimate_signal_2, 0)
      estimate_signal_fusion = signal_fusion(estimate_signal_1)
      L0 = length(estimate_signal_fusion)
      Plots.plot!(collect((4-0.5):(70-0.5)), estimate_signal_fusion[4:70], linewidth = 5,label = "Model",
      xlabel = "Signal intensity of nascent RNA",ylabel = "frequency",
      color = :red)
end

data  = histo_data_vec_normalized[5]
param = params5
compare_plot_G2(data,param)

Plots.savefig("figure0/GAL10_histogram_data5.png")

data  = histo_data_vec_normalized[6]
param = params6
compare_plot_G2(data,param)

Plots.savefig("figure0/GAL10_histogram_data6.png")

data  = histo_data_vec_normalized[7]
param = params7
compare_plot_G2(data,param)
Plots.savefig("figure0/GAL10_histogram_data7.png")

data  = histo_data_vec_normalized[8]
param = params8
compare_plot_G2(data,param)
Plots.savefig("figure0/GAL10_histogram_data8.png")


##
param_G1  = [params1 params2 params3 params4]
fon1 = param_G1[3,:]./(param_G1[3,:] + param_G1[4,:])
burst1 = param_G1[2,:]./param_G1[4,:]
result1 = hcat(param_G1',[burst1 fon1])

param_G2  = [params5 params6 params7 params8]
fon2 = param_G2[3,:]./(param_G2[3,:] + param_G2[4,:])
burst2 = param_G2[2,:]./param_G2[4,:]
result2 = hcat(param_G2',[burst2 fon2])

M1= [mean(result1[:,i]) for i in 1:4]
S1 = [std(result1[:,i]) for i in 1:4]


M2 = [mean(result2[:,i]) for i in 1:6]
S2 = [std(result2[:,i]) for i in 1:6]
