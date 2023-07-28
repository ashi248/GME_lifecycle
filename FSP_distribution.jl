using Flux, DiffEqSensitivity, DifferentialEquations
include("delay_function.jl")

## 2-state gene expression model
# p: parameters of the model
# H: delay function of the model
function model_2state(du,u,p,t,H)
     λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4];
     p2 = p[2] # α and θ
     N = p[3]

     SS = H(t,p2)[1];
     du[1] = -λ1*SS*u[1]+k1*u[2]-k1*u[1];
     du[2] = -λ2*SS*u[2]+k0*u[1]-k0*u[2];
     for i in 1:N
           du[2*i+1] = λ1*SS*(u[2*(i-1)+1]-u[2*i+1])+k0*u[2*i+2]-k1*u[2*i+1];
           du[2*i+2] = λ2*SS*(u[2*(i-1)+2]-u[2*i+2])-k0*u[2*i+2]+k1*u[2*i+1];
     end
end

## FSP method
function FSP_distr(model,p,tspan,saveat)
      λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4]; # parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]# number of product

      u0 = [1;1;zeros(2*N)];
      prob = ODEProblem(model, u0, tspan,p);

      # create data
      t = saveat
      sol = Array(solve(prob, Tsit5(), u0=u0, saveat=t));
      prob_mRNA= sol[collect(1:2:(2*N+1)),:]
      data = convert(Array,prob_mRNA)
      return(data)
end


# gamma delay
function mature_mRNA_distribution(H,L=1,t0=1,r=1,k0=0.5,k1=0.5,λ1 = 40,λ2 = 0,time = 400)
      #L=1;t0=1;k0=0.5;k1=0.5;λ1 = 40;λ2 = 0;
      α = L; θ = t0/L; r = r;
      N = 200;
      p1  = [λ1,λ2,k0,k1]
      p2 = [α,θ,r]
      p3 = N
      p = [p1,p2,p3]

      tspan = (0.0,400.0);
      saveat = [time]
      model(du,u,p,t) = model_2state(du,u,p,t,H)
      probs = FSP_distr(model,p,tspan,saveat)
      return(probs)
end

##
# 4-state model
function model_4general(du,u,p,t,H)
      λ1 = p[1][1];λ2 = p[1][2];λ3 = p[1][3];λ4 = p[1][4];
      k12 = p[1][5];k23 = p[1][6]; k34 = p[1][7];k41 = p[1][8];# parameters for models
      k43 = p[1][9];k32 = p[1][10]; k21 = p[1][11];k14 = p[1][12];# parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]  # number of product

     SS = H(t,p2)[1];
     du[1] = -λ1*SS*u[1]+k14*u[4]+k12*u[2]-(k12+k14)*u[1];
     du[2] = -λ2*SS*u[2]+k21*u[1]+k23*u[3]-(k21+k23)*u[2];
     du[3] = -λ3*SS*u[3]+k32*u[2]+k34*u[4]-(k32+k34)*u[3];
     du[4] = -λ4*SS*u[4]+k41*u[1]+k43*u[3]-(k41+k43)*u[4];
     for i in 1:N
           du[4*i+1] = λ1*SS*(u[4*(i-1)+1]-u[4*i+1])+k14*u[4*i+4]+k12*u[4*i+2]-(k12+k14)*u[4*i+1];;
           du[4*i+2] = λ2*SS*(u[4*(i-1)+2]-u[4*i+2])+k21*u[4*i+1]+k23*u[4*i+3]-(k21+k23)*u[4*i+2];
           du[4*i+3] = λ3*SS*(u[4*(i-1)+3]-u[4*i+3])+k32*u[4*i+2]+k34*u[4*i+4]-(k32+k34)*u[4*i+3];
           du[4*i+4] = λ4*SS*(u[4*(i-1)+4]-u[4*i+4])+k41*u[4*i+1]+k43*u[4*i+3]-(k41+k43)*u[4*i+4];
     end
end


function FSP_distr_4general(model,p,tspan,saveat)

      λ1 = p[1][1];λ2 = p[1][2];λ3 = p[1][3];λ4 = p[1][4];
      k12 = p[1][5];k23 = p[1][6]; k34 = p[1][7];k41 = p[1][8];# parameters for models
      k43 = p[1][9];k32 = p[1][10]; k21 = p[1][11];k14 = p[1][12];# parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]  # number of product

      p0 = [1;1;1;1]
      u0 = [p0;zeros(4*N)];
      prob = ODEProblem(model, u0, tspan,p);

      # create data
      prob_mRNA_1 = zeros(N+1,length(saveat))
      prob_mRNA_2 = zeros(N+1,length(saveat))
      prob_mRNA_3 = zeros(N+1,length(saveat))
      prob_mRNA_4 = zeros(N+1,length(saveat))

      sol = Array(solve(prob, Tsit5(), u0=u0, saveat=saveat));
      for i in 1:length(saveat)
      prob_mRNA_1[:,i]= sol[collect(1:4:(4*N+1)),i]
      prob_mRNA_2[:,i]= sol[collect(2:4:(4*N+2)),i]
      prob_mRNA_3[:,i]= sol[collect(3:4:(4*N+3)),i]
      prob_mRNA_4[:,i]= sol[collect(4:4:(4*N+4)),i]
      end
      prob_1 = convert(Array,prob_mRNA_1)
      prob_2 = convert(Array,prob_mRNA_2)
      prob_3 = convert(Array,prob_mRNA_3)
      prob_4 = convert(Array,prob_mRNA_4)
      return([prob_1,prob_2,prob_3,prob_4])
end
