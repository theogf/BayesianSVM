include("KernelFunctions.jl")


using Distributions

#Objective function for the Gibbs Sampler (to minimize) (see details in Gibbs Sampler notebook), X is actually Xy
function fobj(X, w, Σ, ell = 1)
    return 0.5*dot(w,Σ*w) + 2*sum(broadcast(max, 0.0, ell-X*w))
end;

function GibbsSamplerSVM(X_::Array{Float64,2},y_::Array{Float64,1},γ_::Float64,nepochs_::Int64, burnin_::Int64; ell_::Float64=1.0,
   mcmc_::Bool = true, random_seed_::Int64 = 42, μ_init = 0, print::Bool = true,ϵ_::Float64 = 0.000001,smooth::Int64=10,non_linear::Bool=false,Θ=0,kernel_function=rbf,storing::Bool =false)
   svm_lin(X_,y_,γ_,nepochs_,burnin_,ell=ell_,mcmc=mcmc_,random_seed=random_seed_,ibeta=μ_init,pr=print,ϵ=ϵ_,smoothing_window=smooth,non_linear=non_linear,Θ=Θ,kernel_function=kernel_function,storing=storing)
end

#Algorithm to compute the beta values
function svm_lin(X::Array{Float64,2}, y::Array{Float64,1}, γ::Float64, nepochs::Int64, burnin::Int64; ell::Float64 = 1.0, mcmc::Bool = true,
                  random_seed::Int64 = 123, ibeta = 0, pr::Bool = true,ϵ::Float64 = 0.000001, smoothing_window::Int64 = 10, non_linear::Bool = false, Θ=0, kernel_function=rbf, storing::Bool = false)

          window = 10
          y = vec(y)

          n = size(X,1)
          if non_linear
            k = size(X,1)
          else
            k = size(X,2)
          end
          Σ = γ*eye(k)
          S = eye(k)
          m = ones(k)
          #Initialize a vector using the prior information
          if  ibeta == 0
              ibeta = squeeze(rand(MvNormal(zeros(k),Σ),1),2);
          end
          if non_linear
            if Θ == 0
              Θ = 1.0
            end
            K = (CreateKernelMatrix(X,kernel_function,Θ)+1e8*eye(k))
          end
          #Initializing needed vectors
          β = ibeta/norm(ibeta)
          #Mean beta
          mβ = zeros(k)
          invB = zeros(k,k)
          B = zeros(k,k)
          chain_β = zeros((nepochs-burnin,k)) # in the j-th row is the D-vector beta_j
          chain_mβ = zeros((nepochs-burnin,k))
          chain_invλ = zeros((nepochs-burnin,n))
          fvals = zeros(nepochs)
          mfval = zeros(nepochs-burnin)
          #=====> X is never alone in the algorithm so we can multiply it by Y to save computation time <===#
          if !non_linear
            X = Diagonal(y)*X
          end
          minfval = Inf # minimum value of objective function
          i = 1
          while i < nepochs
              if pr && (i%100)==0
                  println("At iteration $(i)")
              end
              if non_linear
                invλ = sqrt(1.0+2.0/γ)./(abs(ell-y.*β))
                b = γ
              else
                invλ = 1./abs(ell-X*β)
                b = 1
              end
              indinf = isinf(invλ)
              if mcmc
                  for j in length(invλ)
                      invλ[j] = rand(InverseGaussian(invλ[j],b))
                  end
              end
              if non_linear
                #println((trace(S),norm(m)))
                S = Matrix(Symmetric(1/γ*K*inv(K+Diagonal(1./invλ)/γ)*Diagonal(1./invλ),:U))
                m = γ*S*Diagonal(y)*Diagonal(invλ)*(1+1./invλ)
                β = rand(MvNormal(m,S))
              else
                invB=(X.'*Diagonal(invλ)*X+inv(Σ))
                B = Matrix(Symmetric(inv(invB)))
                β = rand(MvNormal(B*X.'*(invλ+1),B))
              end
              if norm(β) > 100
              #    println(β,B,maximum(invλ))
              end
              if storing
                fvals[i] = fobj(X,β,Σ,ell)
              end
              if mcmc && i > burnin
                  mβ = mβ + (β-mβ)./(i-burnin)
                  if storing
                    mfval[i-burnin] = fobj(X,mβ,Σ,ell)
                    chain_β[i-burnin,:] = β
                    chain_invλ[i-burnin,:] = invλ
                  end
                  chain_mβ[i-burnin,:] = mβ
                  smooth_1 = mean(chain_mβ[max(1,i-burnin-2*window):(i-burnin-1),:],1)
                  smooth_2 = mean(chain_mβ[max(2,i-burnin-2*window+1):(i-burnin),:],1)
                  if norm(smooth_1/norm(smooth_1)-smooth_2/norm(smooth_2))<ϵ
                    break;
                  end
              end
              #minfval = min(fvals[i],minfval)
              if !mcmc
                  mβ = β
              end
              i += 1
          end
          println("Gibbs Sampler stopped after $i iterations")
          if storing
            return (mβ,B,chain_β,chain_invλ,fvals,chain_mβ)
          else
            return (mβ,S,chain_mβ)
          end
end


function Henao(X::Array{Float64,2}, y::Array{Float64,1}; kernel=0,γ::Float64=1.0, nepochs::Int64=100,ϵ::Float64 = 1e-5,Θ=[1.0],verbose=false)
  #initialization of parameters
    n = size(X,1)
    k = size(X,1)
    Y = Diagonal(y)
    f = randn(k)
    invλ = abs(rand(k))
    if kernel == 0
      kernel = Kernel("rbf",Θ[1],params=Θ[2])
    elseif typeof(kernel)==AbstractString
      if length(Θ)>1
        kernel = Kernel(kernel,Θ[1],params=Θ[2])
      else
        kernel = Kernel(kernel,Θ[1])
      end
    end
    #Initialize a vector using the prior information
    K = CreateKernelMatrix(X,kernel.compute)
    i = 1
    diff = Inf;
    while i < nepochs && diff > ϵ
        prev_λ = 1./invλ; prev_f = f;
        #Expectation Step
        invλ = sqrt(1.0+2.0/γ)./(abs(1.0-y.*f))
        #Maximization Step
        f = K*inv((K+1.0/γ*diagm(1./invλ)))*Y*(1+1./invλ)
        diff = norm(f-prev_f);
        i += 1
        if verbose
          println("$i : diff = $diff")
        end
    end
    if verbose
      println("Henao stopped after $i iterations")
    end
    return (invλ,K,kernel,y,f)
end

function PredicHenao(X,y,X_test,invλ,K,γ,kernel)
  n = size(X,1)
  n_t = size(X_test,1)
  predic = zeros(n_t)
  Σ = inv(K+1/γ*diagm(1./invλ))
  top = Σ*diagm(y)*(1+1./invλ)
  for i in 1:n_t
    k_star = zeros(n)
    for j in 1:n
      k_star[j] = kernel.compute(X[j,:],X_test[i,:])
    end
    predic[i] = dot(k_star,top)
  end
  return predic
end

function PredictProbaHenao(X,y,X_test,invλ,K,γ,kernel)
  n = size(X,1)
  n_t = size(X_test,1)
  predic = zeros(n_t)
  Σ = inv(K+1/γ*diagm(1./invλ))
  top = Σ*diagm(y)*(1+1./invλ)
  for i in 1:n_t
    k_star = zeros(n)
    for j in 1:n
      k_star[j] = kernel.compute(X[j,:],X_test[i,:])
    end
    k_starstar = kernel.compute(X_test[i,:],X_test[i,:])
    predic[i] = cdf(Normal(),dot(k_star,top)/(1+k_starstar-dot(k_star,Σ*k_star)))
  end
  return predic
end
function FITC_Henao(X::Array{Float64,2}, y::Array{Float64,1}, m::Int64; γ::Float64=1.0, nepochs::Int64=100, pr::Bool = true,ϵ::Float64 = 1e-5, smoothing_window::Int64 = 10, Θ=[1.0,1.0], storing::Bool = false)

          n = size(X,1)
          k = size(X,1)
          Y = Diagonal(y)
          Σ = γ*eye(k)
          f = ones(k)
          kernel = Kernel("rbf",Θ[1],params=Θ[2])
          inducingPoints = KMeansInducingPoints(X,m,10)
          #Initialize a vector using the prior information
          Kmm = (CreateKernelMatrix(inducingPoints,kernel.compute)+γ*eye(m))
          Knm = CreateKernelMatrix(X,kernel.compute,inducingPoints)
          Knn = zeros(k)
          for i in 1:k
            Knn[i] = kernel.compute(X[i,:],X[i,:])
          end
          #minfval = Inf # minimum value of objective function
          i = 1
          println("Running Henao FITC-EC Algorithm")
          while i < nepochs
              if pr && (i%100)==0
                  println("At iteration $(i)")
              end
              #Expectation Step
              invλ = sqrt(1.0+2.0/γ)./(abs(1.0-y.*f))
              #Maximization Step
              f = K*inv((K+1.0/γ*Diagonal(invλ)))*Y*(1+1./invλ)
              i += 1
          end
          println("Gibbs Sampler stopped after $i iterations")
          if storing
            return (mβ,B,chain_β,chain_invλ,fvals,chain_mβ)
          else
            return (mβ,S,chain_mβ)
          end
end

function GibbsNonLinearPredict(β,ζ,X,X_test,kernel_function,Θ)

  n = size(X_test,1)
  ksize = size(X,1)
  predic = zeros(n)
  k_star = zeros(ksize)
  K = CreateKernelMatrix(X,kernel_function,Θ)+1e8*eye(ksize)
  A = inv(K)*ζ*β
  for i in 1:n
    for j in 1:ksize
      k_star[j] = kernelfunction(X_test[i,:],X[j,:],Θ)
    end
    predic[i] = dot(k_star,A)
  end
  return predic
end
