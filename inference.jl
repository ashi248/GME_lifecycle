using DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots
using DSP
include("delay_function.jl")
include("FSP_distribution.jl")


##
function Log_Likelihood_gamma(data, p,samples)
      α = p[1]
      θ = 1 / α
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4]

      p1 = [λ1, λ2, k0, k1]
      p2 = [α, θ]
      p3 = maximum(data.numb)
      p0 = [p1, p2, p3]
      tspan = (0.0, 100.0)
      saveat = [100]
      H = delay_nascent_gamma
      model(du, u, p, t) = model_2state(du, u, p0, t, H)
      prob = FSP_distr(model, p0, tspan, saveat)
      tot_loss = 0
      for i in 1:length(prob)
            if prob[i]>0
                  tot_loss = tot_loss - data.prob[i]*log(prob[i])
            end
      end
      tot_loss = tot_loss*samples
      return (tot_loss)
end

function Log_Likelihood_confidence_gamma(data,samples,p,p0,i)
      p1 = insert!(copy(p),i,p0)
      total = Log_Likelihood_gamma(data, p1,samples)
      total = total
      return(total)
end

function Log_Likelihood_mature(data, p)
      α = p[1]
      θ = p[2]/α
      λ1 = p[3]
      λ2 = 0
      k0 = p[4]
      k1 = p[5]
      p1 = [λ1, λ2, k0, k1]
      p2 = [α, θ, 1.0]
      p3 = length(data) - 1
      p0 = [p1, p2, p3]
      tspan = (0.0, 300.0)
      saveat = [300]
      H = delay_mature_gamma
      model(du, u, p, t) = model_2state(du, u, p0, t, H)
      prob = FSP_distr(model, p0, tspan, saveat)
      tot_loss = -sum(data .* log.(prob))
      return (tot_loss)
end

##
# log likelihood for the experimental datasets
function Log_Likelihood_nuc_cyto_final(data1,data2,p)
      m1 = sum(data1.numb.*data1.prob)
      m2 = sum(data2.numb.*data2.prob)
      rt = 1.98
      α1 = p[1]
      θ1 = rt/α1
      λ1 = m1/1.98/0.17/4
      # λ1 = m1/1.98/0.17
      λ2 = 0
      k0 = p[2]
      k1 = 0.83*k0/0.17
      α2 = 1
      θ2  = 0.85/α2
      p1 = [λ1, λ2, k0, k1]
      p2 = [α1,θ1,θ2]

      p3 = 300
      p0 = [p1, p2, p3]
      tspan = (0.0, 100.0)
      saveat = [100]
      H = delay_nascent_gamma
      model(du, u, p0, t) = model_2state(du, u, p0, t, H)
      pr1 = FSP_distr(model, p0, tspan, saveat)
      pr2 = DSP.conv(pr1,pr1)
      prob1 = DSP.conv(pr2,pr2)
      # prob1 = pr1

      p3 = 150
      p0 = [p1, p2, p3]
      H = delay_mature_gamma
      model(du, u, p0, t) = model_2state(du, u, p0, t, H)
      pr1 = FSP_distr(model, p0, tspan, saveat)
      pr2 = DSP.conv(pr1,pr1)
      prob2 = DSP.conv(pr2,pr2)
      # prob2 = pr1

      tot_loss = 0
      for i in 1:length(data1.prob)
            if prob1[i]>0
                  tot_loss = tot_loss - data1.prob[i]*log(prob1[i])
            end
      end

      for i in 1:length(data2.prob)
            if prob2[i]>0
                  tot_loss = tot_loss - data2.prob[i]*log(prob2[i])
            end
      end
      tot_loss = tot_loss
      return (tot_loss)
end


function Log_Likelihood_nuc_cyto_model0(data1,data2,p)
      m1 = sum(data1.numb.*data1.prob)
      m2 = sum(data2.numb.*data2.prob)
      rt = 1.98
      α1 = p[1]
      θ1 = rt/α1
      λ1 = m1/1.98/0.17/4
      λ2 = 0
      k0 = p[3]
      k1 = 0.83*k0/0.17
      α2 = p[2]
      θ2  = 0.85/α2
      p1 = [λ1, λ2, k0, k1]
      p2 = [α1,θ1,α2,θ2]

      p3 = 300
      p0 = [p1, p2, p3]
      tspan = (0.0, 100.0)
      saveat = [100]
      H = delay_nascent_gamma
      model(du, u, p0, t) = model_2state(du, u, p0, t, H)
      pr1 = FSP_distr(model, p0, tspan, saveat)
      pr2 = DSP.conv(pr1,pr1)
      prob1 = DSP.conv(pr2,pr2)

      p3 = 150
      p0 = [p1, p2, p3]
      H = delay_mature_total
      model(du, u, p0, t) = model_2state(du, u, p0, t, H)
      pr1 = FSP_distr(model, p0, tspan, saveat)
      pr2 = DSP.conv(pr1,pr1)
      prob2 = DSP.conv(pr2,pr2)

      tot_loss = 0
      for i in 1:length(data1.prob)
            if prob1[i]>0
                  tot_loss = tot_loss - data1.prob[i]*log(prob1[i])
            end
      end

      for i in 1:length(data2.prob)
            if prob2[i]>0
                  tot_loss = tot_loss - data2.prob[i]*log(prob2[i])
            end
      end
      tot_loss = tot_loss
      return (tot_loss)
end

##
function Log_Likelihood_confidence(data1,data2,p,p0,i)
      p1 = insert!(copy(p),i,p0)
      total = Log_Likelihood_nuc_cyto_model0(data1,data2,p1)
      total = total
      return(total)
end


## generating test datasets

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

##
# log likelihood for the test datasets
# model using both nuclear and cytoplasmic mRNA
function Log_Likelihood_test_nuc_cyt0(data1,data2,p)
      rt = p[5]
      α1 = p[1]
      θ1 = rt/α1
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4]
      α2 = 1
      θ2 = 1

      p1 = [λ1, λ2, k0, k1]
      p2 = [α1,θ1,α2,θ2]
      p3 = convert(Int,maximum(data1.numb))

      p0 = [p1, p2, p3]
      tspan = (0.0, 100.0)
      saveat = [100]
      H = delay_nascent_gamma
      model(du, u, p0, t) = model_2state(du, u, p0, t, H)
      prob1 = FSP_distr(model, p0, tspan, saveat)

      p3 = convert(Int,maximum(data2.numb))
      p0 = [p1, p2, p3]
      H = delay_mature_gamma
      model(du, u, p0, t) = model_2state(du, u, p0, t, H)
      prob2 = FSP_distr(model, p0, tspan, saveat)

      tot_loss1 = 0
      for i in 1:length(prob1)
            if prob1[i]>0 && (sum(data1.numb .==i-1)==1)
                  prob0 = data1.prob[data1.numb .==i-1][1]
                  tot_loss1 = tot_loss1 - prob0*log(prob1[i])
            end
      end

      tot_loss2 = 0
      for i in 1:length(prob2)
            if prob2[i]>0 && (sum(data2.numb .==i-1)==1)
                  prob0 = data2.prob[data2.numb .== i-1][1]
                  tot_loss2 = tot_loss2 - prob0 *log(prob2[i])
            else
                  tot_loss2 = tot_loss2 - 0
            end
      end

      tot_loss = (tot_loss1+tot_loss2)/10000
      return (tot_loss)
end


function Log_Likelihood_test_nuc_cyt(data1,data2,p,p0)
      p1 = insert!(copy(p),5,p0)
      total = Log_Likelihood_test_nuc_cyt0(data1,data2,p1)
      total = total
      return(total)
end

# model using only nuclear mRNA
function Log_Likelihood_test_nuc0(data1,data2,p)
      rt = p[5]
      α1 = p[1]
      θ1 = rt/α1
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4]
      α2 = 1
      θ2 = 1

      p1 = [λ1, λ2, k0, k1]
      p2 = [α1,θ1,α2,θ2]
      p3 = convert(Int,maximum(data1.numb))

      p0 = [p1, p2, p3]
      tspan = (0.0, 200.0)
      saveat = [200]
      H = delay_nascent_gamma
      model(du, u, p0, t) = model_2state(du, u, p0, t, H)
      prob1 = FSP_distr(model, p0, tspan, saveat)

      tot_loss1 = 0
      for i in 1:length(prob1)
            if prob1[i]>0 && (sum(data1.numb .==i-1)==1)
                  prob0 = data1.prob[data1.numb .==i-1][1]
                  tot_loss1 = tot_loss1 - prob0*log(prob1[i])
            end
      end

      tot_loss = tot_loss1/10000
      return (tot_loss)
end


function Log_Likelihood_test_nuc(data1,data2,p,p0)
      p1 = insert!(copy(p),5,p0)
      total = Log_Likelihood_test_nuc0(data1,data2,p1)
      return(total)
end

##
function Log_Likelihood_nascent_freq(data,p)
      rt = 1
      α1 = p[1]
      θ1 = rt/α1
      λ1 = p[2]
      λ2 = 0
      k0 = p[3]
      k1 = p[4]
      α2 = 1
      θ2 = 1
      p1 = [λ1, λ2, k0, k1]
      p2 = [α1, θ1]
      p3 = convert(Int,maximum(data1.numb))
      p0 = [p1, p2, p3]

      sample_numb = sum(data.numb)
      tspan = (0.0, 100.0)
      saveat = [100]
      H = delay_nascent_gamma
      model(du, u, p, t) = model_2state(du, u, p0, t, H)
      prob = FSP_distr(model, p0, tspan, saveat)

      tot_loss = 0
      for i in 1:length(prob)
            if prob[i]>0 && (sum(data.numb .==i-1)==1)
                  prob0 = data.prob[data.numb .==i-1][1]
                  tot_loss = tot_loss - prob0*log(prob[i])
            end
      end
      tot_loss = tot_loss*samples
      return (tot_loss)

end
