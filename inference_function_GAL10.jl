using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles
using DSP
using SpecialFunctions
using DelaySSAToolkit
using Catalyst
using DiffEqSensitivity, DifferentialEquations
using LinearAlgebra
using StatsBase


##
function model_2state(du, u, p, t)
      α = p[1]
      θ = 0.785 / α
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4]
      N = convert(Int,p[5])
      SS = gamma(α, t/θ)/gamma(α)
      #SS = exp(-t)
      du[1] = -λ1 * SS * u[1] + k0 * u[2] - k1 * u[1]
      du[2] = -λ2 * SS * u[2] + k1 * u[1] - k0 * u[2]
      for i = 1:N
            du[2*i+1] =
                  λ1 * SS * (u[2*(i-1)+1] - u[2*i+1]) + k0 * u[2*i+2] -
                  k1 * u[2*i+1]
            du[2*i+2] =
                  λ2 * SS * (u[2*(i-1)+2] - u[2*i+2]) - k0 * u[2*i+2] +
                  k1 * u[2*i+1]
      end
end


function FSP_distr(model, p, tspan, saveat)
      α = p[1]
      θ = 0.785 / α
      r = 1
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4] # number of product
      N = convert(Int,p[5])
      p0 = k0 / (k1 + k0)
      u0 = [p0; 1 - p0; zeros(2 * N)]
      prob = ODEProblem(model, u0, tspan, p)
      # create data
      t = saveat
      sol = Array(solve(prob, Tsit5(), u0 = u0, saveat = t))
      prob_mRNA = sol[collect(1:2:(2*N+2)), :] + sol[collect(2:2:(2*N+2)), :]
      data = convert(Array, prob_mRNA)
      return (data)
end


function fit_function_G2(p::NamedTuple{(:α, :λ1, :k0, :k1)},numb::Int)
      tspan = (0.0, 100.0)
      saveat = [100]
      p0 = [exp(p.α), exp(p.λ1), exp(p.k0), exp(p.k1),numb]
      model(du, u, p, t) = model_2state(du, u, p0, t)
      prob0 = FSP_distr(model, p0, tspan, saveat)
      prob = DSP.conv(prob0,prob0)
      return (prob)
end

function fit_function(p::NamedTuple{(:α, :λ1, :k0, :k1)},numb::Int)
      tspan = (0.0, 100.0)
      saveat = [100]
      p0 = [exp(p.α), exp(p.λ1), exp(p.k0), exp(p.k1),numb]
      model(du, u, p, t) = model_2state(du, u, p0, t)
      prob = FSP_distr(model, p0, tspan, saveat)
      return (prob)
end

##
function Log_Likelihood_nascent(data,p)

      number = convert.(Int,data.keys)
      M = maximum(number)
      p0 = [exp.(p);M]
      tspan = (0.0, 100.0)
      saveat = [100]
      model(du, u, p, t) = model_2state(du, u, p0, t)
      prob = FSP_distr(model, p0, tspan, saveat)

      prob = abs.(prob)
      prob0 = [prob[i+1] for i in number]
      tot_loss = -sum(data.vals.* log.(prob0))
      return (tot_loss)
end

##
function density_func_uniform(z, Δ::Float64, L1, L2)
    density1(z) = z >= 0 && z <= 1 ? 1 : 0
    density2(z) = z > 1 && z <= 1 + Δ ? 1 / Δ : 0
    L1 / (L1 + L2) * density1(z) .+ L2 / (L1 + L2) * density2(z)
end


function convolve_uniform(param, plotrange::Int; Δ=0.01, kwargs...)
    L1, L2 = param
    num_of_conv = plotrange
    Δx = Δ / 10
    x = Δx:Δx:plotrange
    list_conv = density_func_uniform.(x, Δ, L1, L2)
    list_conv_save = Array{Any,1}(undef, num_of_conv)
    list_conv_save[1] = copy(list_conv)
    for i in 1:num_of_conv-1
        list_conv_save[i+1] = DSP.conv(list_conv_save[i], list_conv * Δx)[1:length(list_conv)]
    end
    base_save = Array{Any,1}(undef, plotrange)
    N = Int(1 / Δx)
    for j in 1:num_of_conv
        base_save[j] = sum.([list_conv_save[j][1+i*N:(1+i)*N] for i in 0:plotrange-1]) * Δx
    end
    base_save
end


function convolve_filter(filter::Matrix, count_data::Vector)
    NT = length(count_data) - 1
    trim_filter = @view filter[1:NT, 1:NT]
    signalv1 = zeros(NT)
    md_ = @view count_data[2:NT+1]
    mul!(signalv1, trim_filter, md_)
    [count_data[1] + signalv1[1]; signalv1[2:end]]
end



function convert_histo(data::Vector)
    # Define histogram edge set (integers)
    max_np = ceil(maximum(data))+1
    min_np = 0
    edge = collect(min_np:1:max_np)
    H = fit(Histogram,data,edge)
    saved=zeros(length(H.weights),2);
    saved[:,1] = edge[1:end-1];
    # Normalize histogram to probability (since bins are defined on integers)
    saved[:,2] = H.weights/length(data);
    return saved[:,1], saved[:,2]
end

##
function fusion(data)
    data_fusion  = copy(data)
    data_fusion[data_fusion .< 3].= 3
    return(data_fusion)
end

function signal_fusion(prob)
    prob_fusion  = copy(prob)
    prob_fusion[4] = sum(prob_fusion[1:3])
    prob_fusion[1:3].=0
    return(prob_fusion)
end
