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
     du[1] = -λ1*SS*u[1]+k0*u[2]-k1*u[1];
     du[2] = -λ2*SS*u[2]+k1*u[1]-k0*u[2];
     for i in 1:N
           du[2*i+1] = λ1*SS*(u[2*(i-1)+1]-u[2*i+1])+k0*u[2*i+2]-k1*u[2*i+1];
           du[2*i+2] = λ2*SS*(u[2*(i-1)+2]-u[2*i+2])-k0*u[2*i+2]+k1*u[2*i+1];
     end
end

function model_2state_1(du,u,p,t,H)
     λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4];
     p2 = p[2] # α and θ
     N = p[3]

     SS = H(t,p2)[1];
     du[1] = -λ1*SS*u[1]+k1*u[2]-k1*u[1];
     du[2] = -λ2*SS*u[2]+k0*u[1]-k0*u[2];
     for i in 1:N
           du[2*i+1] = λ1*SS*(u[2*(i-1)+1]-u[2*i+1])+k1*u[2*i+2]-k1*u[2*i+1];
           du[2*i+2] = λ2*SS*(u[2*(i-1)+2]-u[2*i+2])-k0*u[2*i+2]+k0*u[2*i+1];
     end
end

##
function FSP_distr_time(model,p,tspan,saveat)
      λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4]; # parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]  # number of product

      p0 = k0/(k1+k0)
      u0 = [p0;1-p0;zeros(2*N)];
      prob = ODEProblem(model, u0, tspan,p);

      # create data
      t = saveat
      prob_mRNA = zeros(N+1,length(t))
      sol = Array(solve(prob, Tsit5(), u0=u0, saveat=t));
      for i in 1:length(t)
      prob_mRNA[:,i]= sol[collect(1:2:(2*N+2)),i]+sol[collect(2:2:(2*N+2)),i]
      end
      data = convert(Array,prob_mRNA)
      return(data)
end

## FSP method
function FSP_distr(model,p,tspan,saveat)
      λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4]; # parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]# number of product

      p0 = k0/(k1+k0)
      u0 = [p0;1-p0;zeros(2*N)];
      prob = ODEProblem(model, u0, tspan,p);

      # create data
      t = saveat
      sol = Array(solve(prob, Tsit5(), u0=u0, saveat=t));
      prob_mRNA= sol[collect(1:2:(2*N+1)),:]+sol[collect(2:2:(2*N+2)),:]
      data = convert(Array,prob_mRNA)
      return(data)
end


##
# two copy model
function model_4state(du,u,p,t,H)
      λ1 = p[1][1];λ2 = p[1][2];λ3 = p[1][3];λ4 = p[1][4];
      k0 = p[1][5];k1 = p[1][6]; # parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]  # number of product

     SS = H(t,p2)[1];
     du[1] = -λ1*SS*u[1]+k1*u[4]+k1*u[2]-2*k0*u[1];
     du[2] = -λ2*SS*u[2]+k0*u[1]+k1*u[3]-(k0+k1)*u[2];
     du[3] = -λ3*SS*u[3]+k0*u[2]+k0*u[4]-2*k1*u[3];
     du[4] = -λ4*SS*u[4]+k0*u[1]+k1*u[3]-(k0+k1)*u[4];
     for i in 1:N
           du[4*i+1] = λ1*SS*(u[4*(i-1)+1]-u[4*i+1])+k1*u[4*i+4]+k1*u[4*i+2]-2*k0*u[4*i+1];;
           du[4*i+2] = λ2*SS*(u[4*(i-1)+2]-u[4*i+2])+k0*u[4*i+1]+k1*u[4*i+3]-(k0+k1)*u[4*i+2];
           du[4*i+3] = λ3*SS*(u[4*(i-1)+3]-u[4*i+3])+k0*u[4*i+2]+k0*u[4*i+4]-2*k1*u[4*i+3];
           du[4*i+4] = λ4*SS*(u[4*(i-1)+4]-u[4*i+4])+k0*u[4*i+1]+k1*u[4*i+3]-(k0+k1)*u[4*i+4];
     end
end


function FSP_distr_4state(model,p,tspan,saveat)

      λ1 = p[1][1];λ2 = p[1][2];λ3 = p[1][3];λ4 = p[1][4];
      k0 = p[1][5];k1 = p[1][6]; # parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]  # number of product

      W = [-2*k0   k1      0    k1;
             k0  -(k0+k1)  k1    0;
             0      k0     -2*k1  k0;
              1     1        1    1]
      w0 = [0;0;0;1]

      p0 = W\w0
      #p0 = [0.25;0.25;0.25;0.25]
      u0 = [p0;zeros(4*N)];
      prob = ODEProblem(model, u0, tspan,p);

      # create data
      prob_mRNA = zeros(N+1,length(saveat))
      sol = Array(solve(prob, Tsit5(), u0=u0, saveat=saveat));
      for i in 1:length(saveat)
      prob_mRNA[:,i]= sol[collect(1:4:(4*N+1)),i]+sol[collect(2:4:(4*N+2)),i]+
      sol[collect(3:4:(4*N+3)),i]+sol[collect(4:4:(4*N+4)),i]
      end
      data = convert(Array,prob_mRNA)
      return(data)
end

##

function FSP_distr_1(model,p,tspan,saveat)
      λ1 = p[1][1];λ2 = p[1][2];k0 = p[1][3];k1 = p[1][4]; # parameters for models
      p2 = p[2] # α , θ and r; parameter of delay function
      N = p[3]# number of product

      p0 = k0/(k1+k0)
      #p0 = 0
      #u0 = [p0;1-p0;zeros(2*N)];
      u0 = [1;0;zeros(2*N)];
      prob = ODEProblem(model, u0, tspan,p);

      # create data
      t = saveat
      sol = Array(solve(prob, Tsit5(), u0=u0, saveat=t));
      #prob_mRNA= sol[collect(1:2:(2*N+1)),:]+sol[collect(2:2:(2*N+2)),:]
      #prob_mRNA= sol[collect(1:2:(2*N+1)),:]
      prob_mRNA= sol[collect(2:2:(2*N+2)),:]
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
      probs = FSP_distr_1(model,p,tspan,saveat)
      return(probs)
end
