using Flux, DiffEqSensitivity, DifferentialEquations
using Distributions, Distances
using DelimitedFiles, Plots
using QuadGK
include("delay_function.jl")
include("FSP_distribution.jl")
include("noise.jl")

##
# figure 3A
fc = exp.(collect(range(log(0.01),log(40),50)))
function vary_time(fc,time0)
        mRNA_mean = zeros(length(fc),1)
        mRNA_CV = zeros(length(fc),1)
        mRNA_GC = zeros(length(fc),1)
        for i in 1:length(fc)
                print(i)
                α1 = 1/fc[i]; θ1 = time0/α1;
                θ2 = 1;
                p1  = [40,0,1,1]    #[λ1,λ2,k0,k1]
                p2 = [α1,θ1,θ2]
                H = delay_mature_gamma
                Mean,Var = gamma_distribution_noise(H,p1,p2)
                GC  = gini_coef(p2)
                mRNA_CV[i] = Var/Mean^2
                mRNA_mean[i] = Mean
                mRNA_GC[i] = GC
        end

        α1 = 1; θ1 = 1/α1;
        p1  = [40,0,1,1]     #[λ1,λ2,k0,k1]
        p2 = [α1,θ1]
        H = delay_nascent_gamma
        Mean,Var = gamma_distribution_noise(H,p1,p2)
        mRNA_CV0 = Var/Mean^2
        ratio = mRNA_CV./mRNA_CV0
        return([ratio,mRNA_GC])
end


ratio1,GC1 = vary_time(fc,0.5)
ratio2,GC2 = vary_time(fc,1)
ratio3,GC3 = vary_time(fc,2)

using LaTeXStrings

Plots.plot(fc, ratio1,linewidth=4,xscale =:log,
dpi=600,legendfontsize = 12,tickfontsize = 12,
ylims = [0.5,1.1],labelfontsize=14,size = (500,400),
xlabel = "Noise of elongation",ylabel = "Relative mature mRNA noise",
grid=0,framestyle=:box,line = (:sold),label=L"\mu_\tau=0.5")

Plots.plot!(fc, ratio2,linewidth=4,xscale =:log,ls=:dash,
label=L"\mu_\tau=1")

Plots.plot!(fc, ratio3,linewidth=4,xscale =:log,ls=:dot,
label=L"\mu_\tau=2")

Plots.savefig("figure0/optinal_noise_vary_mu.png")

##

Plots.plot(fc, GC1,linewidth=4,xscale =:log,
dpi=600,legendfontsize = 12,tickfontsize = 12,
labelfontsize=14,size = (500,400),ylims = [0.5,1],
xlabel = "Noise of elongation",ylabel = "Gini coefficient",
grid=0,framestyle=:box,line = (:sold),label=L"\mu_\tau=0.5")

Plots.plot!(fc, GC2,linewidth=4,xscale =:log,ls=:dash,
label=L"\mu_\tau=1")

Plots.plot!(fc, GC3,linewidth=4,xscale =:log,ls=:dot,
label=L"\mu_\tau=2")

Plots.savefig("figure0/gini_vary_mu.png")



##
# figure 3B
fc = exp.(collect(range(log(0.01),log(40),50)))
function vary_rate(fc,rate)
        mRNA_mean = zeros(length(fc),1)
        mRNA_FF = zeros(length(fc),1)
        mRNA_GC = zeros(length(fc),1)
        for i in 1:length(fc)
                print(i)
                α1 = 1/fc[i]; θ1 = 2/α1;
                θ2 = 1;
                p1  = [40,0,1*rate,1*rate]    #[λ1,λ2,k0,k1]
                p2 = [α1,θ1,θ2]
                H = delay_mature_gamma
                Mean,Var = gamma_distribution_noise(H,p1,p2)
                GC  = gini_coef(p2)
                mRNA_FF[i] = Var/Mean^2
                mRNA_mean[i] = Mean
                mRNA_GC[i] = GC
        end

        α1 = 1; θ1 = 1/α1;
        p1  = [40,0,1*rate,1*rate]    #[λ1,λ2,k0,k1]
        p2 = [α1,θ1]
        H = delay_nascent_gamma
        Mean,Var = gamma_distribution_noise(H,p1,p2)
        mRNA_FF0 = Var/Mean^2
        ratio = mRNA_FF./mRNA_FF0
        return([ratio,mRNA_GC])
end

ratio1,GC1 = vary_rate(fc,0.1)
ratio2,GC2 = vary_rate(fc,0.3)
ratio3,GC3 = vary_rate(fc,1)



Plots.plot(fc, ratio1,linewidth=4,xscale =:log,
dpi=600,legendfontsize = 12,tickfontsize = 12,
ylims = [0.5,1.12],labelfontsize=14,size = (500,400),
xlabel = "Noise of elongation",ylabel = "Relative mature mRNA noise",
grid=0,framestyle=:box,line = (:sold),label=L"k_{promoter}=0.1")
Plots.plot!(fc, ratio2,linewidth=4,xscale =:log,label=L"k_{promoter}=0.5",
ls=:dash)
Plots.plot!(fc, ratio3,linewidth=4,xscale =:log,label=L"k_{promoter}=1",
ls=:dot)

Plots.savefig("figure0/optinal_noise_vary_k.png")

##

Plots.plot(fc, GC1,linewidth=4,xscale =:log,
dpi=600,legendfontsize = 12,tickfontsize = 12,
ylims = [0.5,1.1],labelfontsize=14,size = (500,400),
xlabel = "Noise of elongation",ylabel = "Gini coefficient",
grid=0,framestyle=:box,line = (:sold),label=L"k_{promoter}=0.1")

Plots.plot!(fc, GC2,linewidth=4,xscale =:log,label=L"k_{promoter}=0.5",ls=:dash)

Plots.plot!(fc, GC3,linewidth=4,xscale =:log,label=L"k_{promoter}=1",ls=:dot)

Plots.savefig("figure0/gini_vary_k.png")
