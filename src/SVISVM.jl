#module SVISVM
include("KernelFunctions.jl")
include("AFKMC2.jl")
using KernelFunctions
using Distributions
using StatsBase
using PyPlot
using GaussianMixtures
using ScikitLearn
using ScikitLearn: fit!

#export VariationalInferenceAlgorithm
#export StochasticVariationalInferenceAlgorithm
#export PredictiveDistribution

#Corresponds to the SVISVM model
type VariationalInferenceSVM
  #Stochastic parameters
  Stochastic::Bool
    nSamplesUsed::Int64 #Size of the minibatch used
    κ_s::Float64 #Parameters for decay of learning rate (iter + κ)^-τ in case adaptative learning rate is not used
    τ_s::Float64
  #Non linear parameters
  NonLinear::Bool
  Sparse::Bool
    kernels::Array{Kernel,1} #Kernels function used
    γ::Float64 # Regularization parameter of the noise
    m::Int64 #Number of inducing points
    inducingPointsSelectionMethod::String #Way to select the inducing points ("Random","KMeans","GMM")
  #Autotuning parameters
  Autotuning::Bool
    κ_Θ::Float64 #Parameters for decay of learning rate for the hyperparameter  (iter + κ)^-τ
    τ_Θ::Int64
    autotuningFrequency::Int64 #Frequency of update of the hyperparameter
  #Flag for adaptative learning rate for the SVI
  AdaptativeLearningRate::Bool
  #General Parameters for training
  Intercept::Bool
  ϵ::Float64 #Desired precision (on ||β(t)-β(t-1)||)
  nEpochs::Int64 #Maximum number of iterations
  β_init::Array{Float64,1} #Initial value for β
  smoothingWindow::Int64
  VerboseLevel::Int64
  Storing::Bool
    StoringFrequency::Int64
    storedValues::Tuple


  #Functions
  Kernel_function::Function
  Predict::Function
  PredictProba::Function
  ELBO::Function
  dELBO::Function
  Plotting::Function
  Update::Function

  #Parameters learned with training
  nSamples::Int64 # Number of data points
  nFeatures::Int64 # Number of features
  μ::Array{Float64,1} # Mean for variational distribution
  η_1::Array{Float64,1} #Natural Parameter #1
  ζ::Array{Float64,2} # Covariance matrix of variational distribution
  η_2::Array{Float64,2} #Natural Parameter #2
  α::Array{Float64,1} # Distribution parameter of the GIG distribution of the latent variables
  invΣ::Array{Float64,2} #Inverse Prior Matrix for the linear case
  invK::Array{Float64,2} #Inverse Kernel Matrix for the nonlinear case
  invKmm::Array{Float64,2} #Inverse Kernel matrix of inducing points
  Ktilde::Array{Float64,1} #Diagonal of the covariance matrix between inducing points and generative points
  κ::Array{Float64,2} #Kmn*invKmm
  inducingPoints::Array{Float64,2} #Inducing points coordinates for the Big Data GP
  top #Storing matrices for repeated predictions (top and down are numerator and discriminator)
  down
  MatricesPrecomputed::Bool #Flag to know if matrices needed for predictions are already computed or not
  ρ_s::Float64 #Learning rate for CAVI
  g::Array{Float64,1} # g & h are expected gradient value for computing the adaptive learning rate
  h::Float64
  ρ_Θ::Float64 # learning rate for auto tuning
  initialized::Bool
  StoredValues::Array{Float64,2}
  StoreddELBO::Array{Float64,2}
  evol_β::Array{Float64,2}


  #Constructor
  function VariationalInferenceSVM(Stochastic::Bool;
                                  Sparse::Bool=false,NonLinear::Bool=true,AdaptativeLearningRate::Bool=true,Autotuning::Bool=false,
                                  nEpochs::Int64 = 2000,
                                  batchSize::Int64=-1,κ_s::Float64=1.0,τ_s::Int64=100,
                                  kernels=0,γ::Float64=1e-3,m::Int64=100,inducingPointsSelectionMethod::String="Random",κ_Θ::Float64=1.0,τ_Θ::Int64=100,autotuningfrequency::Int64=10,
                                  Intercept::Bool=false,ϵ::Float64=1e-5,β_init=[0.0],smoothingWindow::Int64=10,
                                  Storing::Bool=false,StoringFrequency::Int64=1,VerboseLevel::Int64=0)
    iter = 1
    if kernels == 0
      kernels = [Kernel("rbf",1.0,params=1.0)]
    end
    this = new(Stochastic,batchSize,κ_s,τ_s,NonLinear,Sparse,kernels,γ,m,inducingPointsSelectionMethod,Autotuning,κ_Θ,τ_Θ,autotuningfrequency,AdaptativeLearningRate,Intercept,ϵ,nEpochs,β_init,smoothingWindow,VerboseLevel,Storing,StoringFrequency)
    this.initialized = false
    if NonLinear
      this.top = 0
      this.down = 0
      MatricesPrecomputed = false
      this.Kernel_function = function(X1,X2)
          dist = 0
          for i in 1:size(this.kernels,1)
            dist += this.kernels[i].coeff*this.kernels[i].compute(X1,X2)
          end
          return dist
      end

      if Sparse
        this.Predict = function(X,X_test)
            SparsePredict(X_test,this)
          end
        this.PredictProba = function(X,X_test)
            SparsePredictProb(X_test,this)
          end
        this.ELBO = function(X,y)
            SparseELBO(this,y)
          end
          this.dELBO = function(X,y) #Not correct to change later
              return 0
          end
      else
        this.Predict = function(X,X_test)
            NonLinearPredict(X,X_test,this)
          end
        this.PredictProba = function(X,X_test)
            NonLinearPredictProb(X,X_test,this)
          end
        this.ELBO = function(X,y)
            ELBO_NL(this,y)
          end
        this.dELBO = function(X,y)
            dELBO_NL(y,this.μ,this.ζ,this.α,this.invK,this.Autotuning ? this.J : eye(size(X,1)))
          end
      end

    else
      this.Predict = function(X)
          LinearPredict(X,this.μ)
        end
      this.PredictProba = function(X)
          LinearPredictProb(X,this.μ,this.ζ)
        end
      this.ELBO = function(X,y;precomputed::Bool=true)
          ELBO(Diagonal(y)*X,this.μ,this.ζ,this.α,inv(this.invΣ))
        end
      this.dELBO = function(X,y;precomputed::Bool=true)
          dELBO(Diagonal(y)*X,this.μ,this.ζ,this.α,inv(this.invΣ))
        end
    end
    this.Plotting = function(s::String)
        Plotting(s,this)
      end
    this.Update = function(X::Array{Float64,2},y::Array{Float64,1},iter)
        Update(this,X,y,iter)
      end
    return this
  end
  #end of constructor
end

#Function to check consistency of the different parameters and the possible correction of some of them in some cases
function ModelVerification(model::VariationalInferenceSVM,XSize,ySize)
  if model.Intercept && model.NonLinear
    warn("Not possible to have intercept for the non linear case, removing automatically this option")
    model.Intercept = false
  end
  if model.Sparse && !model.NonLinear
    warn("Model cannot be sparse and linear at the same time, assuming linear model")
  end
  if model.NonLinear && model.Sparse
    if model.m > XSize[1]
      warn("There are more inducing points than actual points, setting it to 10%")
      model.m = XSize[1]÷10
    end
  end
  if XSize[1] != ySize[1]
    warn("There is a dimension problem with the data size(y) != size(X)")
    return false
  end
  if model.γ <= 0
    warn("Gamma should be strictly positive, setting it to default value 1.0e-3")
    model.γ = 1e-3
  end
  if model.nSamplesUsed == -1 && model.Stochastic
    warn("No batch size has been given, stochastic option has been removed")
    model.Stochastic = false
  end
  return true
end

function TrainVISVM(model::VariationalInferenceSVM,X::Array{Float64,2},y::Array{Float64,1})
  #Verification of consistency of the model
  if !ModelVerification(model,size(X),size(y))
    return
  end
  model.nSamples = length(y)
  model.nFeatures = model.NonLinear ? (model.Sparse ? model.m : length(y)) : size(X,2)

  if model.VerboseLevel > 0
    println("Starting training of data of size $((model.nSamples,size(X,2))), using the"*(model.Autotuning ? " autotuned" : "")*(model.Stochastic ? " stochastic" : "")*(model.NonLinear ? " kernel" : " linear")*" method"
    *(model.AdaptativeLearningRate ? " with adaptative learning rate" : "")*(model.Sparse ? " with inducing points" : ""))
  end

  #Initialization of the variables
  if !model.initialized
    if model.β_init[1] == 0 || length(model.β_init) != nFeatures
      if model.VerboseLevel > 1
        warn("Initial vector is sampled from a multinormal distribution")
      end
      model.μ = randn(model.nFeatures)
    else
      model.μ = model.β_init
    end
    if model.Intercept
      model.nFeatures += 1
      X = [ones(Float64,model.nFeatures) X]
    end
    #Necessary to initialize only for first computation of the ELBO
    model.α = abs(rand(model.nSamples))
    model.ζ = eye(model.nFeatures)
    #Creation of the Kernel Matrix and its inverse in the different cases as well as the prior
    if model.NonLinear
      if !model.Sparse
        model.invK = inv(Symmetric(CreateKernelMatrix(X,model.Kernel_function) + model.γ*eye(model.nFeatures),:U))
      end
      if model.Sparse
        if model.inducingPointsSelectionMethod == "Random"
          model.inducingPoints = X[StatsBase.sample(1:model.nSamples,model.m,replace=false),:]
        elseif model.inducingPointsSelectionMethod == "KMeans"
          model.inducingPoints = KMeansInducingPoints(X,model.m,10)
        elseif model.inducingPointsSelectionMethod == "GMM"
          model.inducingPoints = (ScikitLearn.fit!(GMM(n_components = model.m),X)).μ
        end

        model.invKmm = Matrix(Symmetric(inv(CreateKernelMatrix(model.inducingPoints,model.Kernel_function)+model.γ*eye(model.nFeatures))))
        Knm = CreateKernelMatrix(X,model.Kernel_function,X2=model.inducingPoints)
        model.κ = Knm*model.invKmm
        model.Ktilde = CreateDiagonalKernelMatrix(X,model.Kernel_function) + model.γ*ones(size(X,1)) - squeeze(sum(model.κ.*Knm,2),2) #diag(model.κ*transpose(Knm))
      end
    elseif !model.NonLinear
      model.invΣ =  (1.0/model.γ)*eye(model.nFeatures)
    end
    if (model.nSamplesUsed <= 0 || model.nSamplesUsed > model.nSamples)
      model.nSamplesUsed = model.nSamples
    end
    #Initialization of the natural parameters
    model.η_2 = -0.5*inv(model.ζ)
    model.η_1 = 2*model.η_2*model.μ
    if model.AdaptativeLearningRate && model.Stochastic
      batchindices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false)
      Z = model.NonLinear ? Diagonal(y) : Diagonal(y)*X
      (grad_1,grad_2) = NaturalGradientELBO(model.α,(model.NonLinear && model.Sparse) ? Z*model.κ : Z, model.NonLinear ? (model.Sparse ? model.invKmm : model.invK) : model.invΣ,model.nSamples/model.nSamplesUsed)
      model.τ_s = model.nSamplesUsed
      model.g = vcat(grad_1,reshape(grad_2,size(grad_2,1)^2))
      model.h = norm(vcat(grad_1,reshape(grad_2,size(grad_2,1)^2)))^2
    end
    model.ρ_Θ = model.Autotuning? (1+model.τ_Θ)^(-model.κ_Θ) : 1.0;
    model.ρ_s = model.Stochastic ? (model.AdaptativeLearningRate ? dot(model.g,model.g)/model.h : (1+model.τ_s)^(-model.κ_s)) : 1.0
    if model.Storing
      # Storing trace(ζ),ELBO,max(|α|),ρ_s,ρ_Θ/ρ_γ,||Θ||/γ
      model.StoredValues = zeros(model.nEpochs,6)
      model.StoreddELBO = zeros(model.nEpochs,4)
      model.StoredValues[1,:] = [trace(model.ζ),model.ELBO(X,y),maximum(abs(model.α)),model.ρ_s,model.γ,model.Autotuning ? model.ρ_Θ : 0.0]
      model.StoreddELBO[1,:] = model.dELBO(X,y)
    end
    model.initialized = true
    model.down = 0
    model.top = 0
    model.MatricesPrecomputed = false
  end
  evol_β = zeros(model.nEpochs,model.nFeatures)
  evol_β[1,:] = model.μ

  batchindices = collect(1:model.nSamples)
  prev = 0
  current = 0
  if model.VerboseLevel > 2 || (model.VerboseLevel > 1)
    current = model.ELBO(X,y)
  end
  conv = Inf #Initialization of the Convergence value
  iter::Int64 = 1
  ##End of Initialization of the parameters
  if model.VerboseLevel > 1
    println("Iteration $iter / $(model.nEpochs) (max)")
    println("Convergence : $conv, ELBO : $current")
  end
  #Two criterions for stopping, number of iterations or convergence
  while iter < model.nEpochs && conv > model.ϵ
    #Print some of the parameters
    model.Update(X,y,iter)
    iter += 1
    evol_β[iter,:] = model.μ
    #smooth_1 = mean(evol_β[max(1,iter-2*model.smoothingWindow):iter-1,:],1);smooth_2 = mean(evol_β[max(2,iter-2*model.smoothingWindow+1):iter,:],1);
    smooth_1 = mean(evol_β[max(1,iter-2*model.smoothingWindow):iter-1,:],1);smooth_2 = mean(evol_β[max(2,iter-2*model.smoothingWindow+1):iter,:],1);
    conv = norm(smooth_1/norm(smooth_1)-smooth_2/norm(smooth_2))
    #prev = current
    if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
      current = model.ELBO(X,y)
    end
    #conv = abs(current-prev)
    if model.Storing && iter%model.StoringFrequency == 0
      if model.NonLinear && model.Stochastic && ((!model.Autotuning && iter<=2) || (model.Autotuning && ((iter-1)%model.autotuningFrequency == 0)))
        println("Recomputing Kernel matrices")
        model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(X,model.Kernel_function)+model.γ*eye(model.nSamples)),:U))
        if model.Autotuning
          model.J = CreateKernelMatrix(X,deriv_rbf,model.Θ)
        end
      end
      model.StoredValues[iter÷model.StoringFrequency,:] = [trace(model.ζ),model.ELBO(X,y),maximum(abs(model.α)),model.ρ_s,model.γ,model.Autotuning ? model.ρ_Θ : 0.0,]
      model.StoreddELBO[iter÷model.StoringFrequency,:] = model.dELBO(X,y)
      #println(model.StoreddELBO[iter÷model.StoringFrequency,:])
    end
    if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
      println("Iteration $iter / $(model.nEpochs) (max)")
      println("Convergence : $conv, ELBO : $current")
      if model.Autotuning
        println("Gamma : $(model.γ)")
        for i in 1:size(model.kernels,1)
          println("(Coeff,Parameter) for kernel $i : $((model.kernels[i].coeff,(model.kernels[i].Nparams > 0)? model.kernels[i].param : 0))")
        end
        println("rho theta : $(model.ρ_Θ)")
      end
    end
  end
  if model.VerboseLevel > 0
    println("Training ended after $iter iterations")
  end
  if model.Storing
    model.StoredValues = model.StoredValues[1:iter÷model.StoringFrequency,:];
    model.StoreddELBO = model.StoreddELBO[1:iter÷model.StoringFrequency,:];
    model.evol_β = evol_β[1:iter,:]
  end
  return model
end



function Update(model::VariationalInferenceSVM,X::Array{Float64,2},y::Array{Float64,1},iter::Int64) #Coordinates ascent of the parameters
    if model.Stochastic
      batchindices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false)
    else
      batchindices = collect(1:model.nSamples)
    end
    model.top = 0; model.down = 0; model.MatricesPrecomputed = false;#Need to recompute the matrices
    #Definition of the Z matrix, different for everycase
    Z = model.NonLinear ? (model.Sparse ? Diagonal(y[batchindices])*model.κ[batchindices,:] : Diagonal(y[batchindices]) ) : Diagonal(y[batchindices])*X[batchindices,:]
    #Computation of latent variables
    model.α[batchindices] = (1 - Z*model.μ).^2 + squeeze(sum((Z*model.ζ).*Z,2),2)
    if model.Sparse && model.NonLinear
      model.α[batchindices] += model.Ktilde[batchindices] #Cf derivation of updates
    end

    #Compute the natural gradient
    (grad_η_1,grad_η_2) = NaturalGradientELBO(model.α[batchindices],Z, model.NonLinear ? (model.Sparse ? model.invKmm : model.invK) : model.invΣ, model.Stochastic ? model.nSamples/model.nSamplesUsed : 1.0)

    #Compute the learning rate
    if model.AdaptativeLearningRate && model.Stochastic
      #Using the paper on the adaptive learning rate for the SVI (update from the natural gradients)
      model.g = (1-1/model.τ_s)*model.g + vcat(grad_η_1-model.η_1,reshape(grad_η_2-model.η_2,size(grad_η_2,1)^2))./model.τ_s
      model.h = (1-1/model.τ_s)*model.h +norm(vcat(grad_η_1-model.η_1,reshape(grad_η_2-model.η_2,size(grad_η_2,1)^2)))^2/model.τ_s
      model.ρ_s = norm(model.g)^2/model.h
      #if iter%1==0
      #  println("g : $(norm(model.g)^2), h : $(model.h), rho : $(model.ρ_s), tau : $(model.τ_s)")
      #end
      model.τ_s = (1.0 - model.ρ_s)*model.τ_s + 1.0
    elseif model.Stochastic
      #Simple model of learning rate
      model.ρ_s = (iter+model.τ_s)^(-model.κ_s)
    else
      #Non-Stochastic case
      model.ρ_s = 1.0
    end
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)

    #Autotuning part, only happens every $autotuningFrequency iterations
    if model.Autotuning && (iter%model.autotuningFrequency == 0)
      if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
        println("Before hyperparameter optimization ELBO = $(model.ELBO(X,y))")
      end
      model.ρ_Θ = (iter+model.τ_Θ)^(-model.κ_Θ)
      if model.NonLinear
        if model.Sparse
          update_hyperparameter_Sparse!(model,X,y)
          model.invKmm = Matrix(Symmetric(inv(CreateKernelMatrix(model.inducingPoints,model.Kernel_function)+model.γ*eye(model.m)),:U))
          model.κ = CreateKernelMatrix(X,model.Kernel_function,X2=model.inducingPoints)*model.invKmm
        else
          update_hyperparameter_NL!(model,X,y)
          model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(X,model.Kernel_function)+model.γ*eye(size(X,1))),:U))
        end
      else
        model.γ = update_hyperparameter(model)
        model.invΣ = (1/model.γ)*eye(model.nFeatures)
      end
      if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%100==0)
        println("After hyperparameter optimization ELBO = $(model.ELBO(X,y))")
      end
    end
end;

function NaturalGradientELBO(α,Z,invPrior,stoch_coef)
  grad_1 =  stoch_coef*transpose(Z)*(1./sqrt(α)+1)
  grad_2 = -0.5*(stoch_coef*transpose(Z)*Diagonal(1./sqrt(α))*Z + invPrior)
  (grad_1,grad_2)
end

function update_hyperparameter!(model) #Gradient ascent for γ, noise
    model.γ = model.γ + model.ρ_Θ*0.5*((trace(model.ζ)+norm(model.μ))/(model.γ^2.0)-model.nFeatures/model.γ)
end

function update_hyperparameter_NL!(model,X,y)#Gradient ascent for Θ , kernel parameters
    if model.invK == 0
      model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(X,model.Kernel_function)+model.γ*eye(size(X,1))),:U))
    end
    NKernels = size(model.kernels,1)
    A = model.invK*model.ζ-eye(model.nFeatures)
    grad_γ = 0.5*(sum(model.invK.*A)+dot(model.μ,model.invK*model.invK*model.μ))
    if model.VerboseLevel > 2
      println("Grad gamma : $grad_γ")
    end
    model.γ = ((model.γ + model.ρ_Θ*grad_γ) < 0 ) ? model.γ/2 : (model.γ + model.ρ_Θ*grad_γ)
    #Update of both the coefficients and hyperparameters of the kernels
    if NKernels > 1 #If multiple kernels only update the kernel weight
      for i in 1:NKernels
        V = model.invK.*CreateKernelMatrix(X,model.kernels[i].compute)
        grad =  0.5*(sum(V.*A)+dot(model.μ,V*model.invK*model.μ))#update of the coeff
        if model.VerboseLevel > 2
          println("Grad kernel $i: $grad")
        end
        model.kernels[i].coeff =  ((model.kernels[i].coeff + model.ρ_Θ*grad) < 0 ) ? model.kernels[i].coeff/2 : (model.kernels[i].coeff + model.ρ_Θ*grad)
      end
    elseif model.kernels[1].Nparams > 0 #If only one update the kernel lengthscale
        V = model.invK*model.kernels[1].coeff*CreateKernelMatrix(X,model.kernels[1].compute_deriv)
        grad =  0.5*(sum(V.*A)+dot(model.μ,V*model.invK*model.μ))#update of the hyperparameter
        model.kernels[1].param =  ((model.kernels[1].param + model.ρ_Θ*grad) < 0 ) ? model.kernels[1].param/2 : (model.kernels[1].param + model.ρ_Θ*grad)
        if model.VerboseLevel > 2
          println("Grad kernel: $grad")
        end
    end
end

function update_hyperparameter_Sparse!(model,X,y)#Gradient ascent for Θ , kernel parameters #Not finished !!!!!!!!!!!!!!!!!!!!!!!!!!
  NKernels = size(model.kernels,1)
  A = eye(model.nFeatures)-model.invKmm*model.ζ
  B = model.μ*transpose(model.μ) + model.ζ
  Kmn = CreateKernelMatrix(model.inducingPoints,model.Kernel_function;X2=X)
  #Computation of noise constant
  if model.inducingPointsSelectionMethod == "Random"
    Jnm = CreateKernelMatrix(X,delta_kroenecker,X2=model.inducingPoints)
  else
    Jnm = 0
  end
  ι = (Jnm-model.κ)*model.invKmm
  grad_γ = -0.5*(sum(model.invKmm.*A) - dot(model.μ, transpose(model.μ)*model.invKmm*model.invKmm + 2*transpose(ones(size(X,1))+1./sqrt(model.α))*diagm(y)*ι)+
  dot(1./sqrt(model.α),diag(model.κ*(B*transpose(ι)-transpose(Jnm)) + ι*(B*transpose(model.κ)-Kmn) )+ ones(size(X,1))))
  if model.VerboseLevel > 2
    println("Grad gamma : $grad_γ")
  end
  #model.γ = ((model.γ + model.ρ_Θ*grad_γ) < 0 ) ? (model.γ < 1e-7 ? model.γ : model.γ/2) : (model.γ + model.ρ_Θ*grad_γ)
  if NKernels > 1
    for i in 1:NKernels
      Jnm = CreateKernelMatrix(X,model.kernels[i].compute,X2=model.inducingPoints)
      Jnn = CreateDiagonalKernelMatrix(X,model.kernels[i].compute)
      Jmm = CreateKernelMatrix(model.inducingPoints,model.kernels[i].compute)
      ι = (Jnm-model.κ*Jmm)*model.invKmm
      V = model.invKmm*Jmm
      grad = -0.5*(sum(V.*A) - dot(model.μ, transpose(model.μ)*V*model.invKmm + 2*transpose(ones(size(X,1))+1./sqrt(model.α))*diagm(y)*ι) +
      dot(1./sqrt(model.α),diag(model.κ*(B*transpose(ι)-transpose(Jnm)) + ι*(B*transpose(model.κ)-Kmn))+ Jnn))
      model.kernels[i].coeff =  ((model.kernels[i].coeff + model.ρ_Θ*grad) < 0 ) ? model.kernels[i].coeff/2 : (model.kernels[i].coeff + model.ρ_Θ*grad)
      if model.VerboseLevel > 2
        println("Grad kernel $i: $grad")
      end
    end
  elseif model.kernels[1].Nparams > 0 #Update of the hyperparameters of the KernelMatrix
    Jnm = model.kernels[1].coeff*CreateKernelMatrix(X,model.kernels[1].compute_deriv,X2=model.inducingPoints)
    Jnn = model.kernels[1].coeff*CreateDiagonalKernelMatrix(X,model.kernels[1].compute_deriv)
    Jmm = model.kernels[1].coeff*CreateKernelMatrix(model.inducingPoints,model.kernels[1].compute_deriv)
    ι = (Jnm-model.κ*Jmm)*model.invKmm
    V = model.invKmm*Jmm
    grad = -0.5*(sum(V.*A) - dot(model.μ, transpose(model.μ)*V*model.invKmm + 2*transpose(ones(size(X,1))+1./sqrt(model.α))*diagm(y)*ι) +
    dot(1./sqrt(model.α),diag(model.κ*(B*transpose(ι)-transpose(Jnm)) + ι*(B*transpose(model.κ)-Kmn))+Jnn))
    model.kernels[1].param =  ((model.kernels[1].param + model.ρ_Θ*grad) < 0 ) ? model.kernels[1].param/2 : (model.kernels[1].param + model.ρ_Θ*grad)
    if model.VerboseLevel > 2
      println("Grad kernel: $grad, new param is $(model.kernels[1].param)")
    end
  end
end

function ELBO(Z,μ,ζ,α,Σ) #Compute the loglikelihood of the training data, ####-----Could be improved in algebraic form---####
    n = size(Z,1)
    likelihood = getindex(0.5*(logdet(ζ)-logdet(Σ)-trace(inv(Σ)*ζ)-transpose(μ)*ζ*μ),1) + n*(log(2)-0.5*log(2*pi)-1);
    for i in 1:n
        likelihood += 1.0/2.0*log(α[i]) + log(besselk(0.5,α[i])) + dot(vec(Z[i,:]),μ) + getindex(0.5/α[i]*(α[i]^2-(1-dot(vec(Z[i,:]),μ))^2 - transpose(vec(Z[i,:]))*ζ*vec(Z[i,:])))
    end
    return likelihood
end;

function dELBO(Z,μ,ζ,α,Σ)
    (n,p) = size(Z)
    dζ = 0.5*(inv(ζ)-inv(Σ)-transpose(Z)*Diagonal(1./sqrt(α))*Z)
    dμ = -inv(ζ)*μ + transpose(Z)*(1./sqrt(α)+1)
    dα = zeros(n)
    for i in 1:n
      dα[i] = ((1-dot(Z[i,:],μ))^2 + dot(Z[i,:],ζ*Z[i,:]))/(2*(α[i])) - 0.5
    end
    γ = Σ[1,1]
    dγ = 0.5*((trace(ζ)+norm(μ))/(γ^2.0)-p/γ)
    return [trace(dζ),norm(dμ),norm(dα),dγ]
end


function ELBO_NL(model,y)
  n = size(y,1)
  ELBO = 0.5*(logdet(model.ζ)+logdet(model.invK)-trace(model.invK*model.ζ)-dot(model.μ,model.invK*model.μ))
  for i in 1:n
    ELBO += 0.25*log(model.α[i])+log(besselk(0.5,sqrt(model.α[i])))+y[i]*model.μ[i]+(model.α[i]-(1-model.y[i]*model.μ[i])^2-model.ζ[i,i])/(2*sqrt(model.α[i]))
  end
  return ELBO
end

function dELBO_NL(y,μ,ζ,α,invK,J)
  n = size(X,1)
  dζ = 0.5*(inv(ζ)-invK-Diagonal(1./sqrt(α)))
  dμ = -inv(ζ)*μ + Diagonal(y)*(1./sqrt(α)+1)
  dα = zeros(n)
  for i in 1:n
    dα[i] = ((1-y[i]*μ[i])^2+ζ[i,i])/(2*α[i])-0.5
  end
  dΘ = 0.5*(trace(invK*J*(invK*ζ-1))+dot(μ,invK*J*invK*μ))
  return [trace(dζ),norm(dμ),norm(dα),norm(dΘ)]
end

function SparseELBO(model,y)
  ELBO = 0.0
  ELBO += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
  ELBO += -0.5*(sum(model.invKmm.*model.ζ)+dot(model.μ,model.invKmm*model.μ)) #trace replaced by sum
  ELBO += dot(y,model.κ*model.μ)
  ELBO += sum(0.25*log(model.α) + log(besselk(0.5,sqrt(model.α))))
  ζtilde = model.κ*model.ζ*transpose(model.κ)
  for i in 1:length(y)
    ELBO += 0.5/sqrt(model.α[i])*(model.α[i]-(1-y[i]*dot(model.κ[i,:],model.μ))^2-(ζtilde[i,i]+model.Ktilde[i]))
  end
  return ELBO
end


function Plotting(option::String,model::VariationalInferenceSVM)
  if !model.Storing
    warn("Data was not saved during training, please rerun training with option Storing=true")
    return
  elseif isempty(model.StoredValues )
    warn("Model was not trained yet, please run a dataset before");
    return
  end
  figure("Evolution of model properties over time");
  iterations = collect(1:size(model.evol_β,1))
  sparseiterations = collect(linspace(1,size(model.evol_β,1),size(model.StoredValues,1)))
  if option == "All"
    nFeatures = model.Autotuning ? 6 : 4;
    subplot(nFeatures÷2,2,1)
    plot(sparseiterations,model.StoredValues[:,1])
    ylabel(L"Trace($\zeta$)")
    subplot(nFeatures÷2,2,2)
    plot(iterations,sqrt(sumabs2(model.evol_β,2)))
    ylabel(L"Normalized $\beta$")
    subplot(nFeatures÷2,2,3)
    plot(sparseiterations,model.StoredValues[:,2])
    ylabel("ELBO")
    subplot(nFeatures÷2,2,4)
    semilogy(sparseiterations,model.StoredValues[:,4])
    ylabel(L"\rho_s")
    if model.Autotuning
      subplot(nFeatures÷2,2,5)
      semilogy(sparseiterations,model.StoredValues[:,5])
      ylabel(model.NonLinear ? L"||\theta||" : L"\gamma")
      subplot(nFeatures÷2,2,6)
      semilogy(sparseiterations,model.StoredValues[:,6])
      ylabel(L"\rho_\theta")
    end
  elseif option == "dELBO"
    (nIterations,nFeatures) = size(model.StoreddELBO)
    DerivativesLabels = ["d\zeta" "d\mu" "d\alpha" "d\theta"]
    for i in 1:4
      if i <= 3  || (i==4 && model.Autotuning)
        subplot(2,2,i)
        plot(sparseiterations,model.StoreddELBO[:,i])
        ylabel(DerivativesLabels[i])
      end
    end
  elseif option == "Beta"
    plot(iterations,sqrt(sumabs2(model.evol_β)))
    ylabel(L"Normalized $\beta$")
    xlabel("Iterations")
  elseif option == "ELBO"
    plot(iterations,model.StoredValues[:,2])
    ylabel("ELBO")
    xlabel("Iterations")
  else
    warn("Option not available, chose among those : All, dELBO, Beta, Autotuning, ELBO")
  end
  return;
end


function LinearPredict(X,β::Array{Float64,1})
  return X*β
end

function LinearPredictProb(X,β::Array{Float64,1},ζ::Array{Float64,2})
  n = size(X,1)
  predic = zeros(n)
  for i in 1:n
    predic[i] = cdf(Normal(),(dot(X[i,:],β))/(dot(X[i,:],ζ*X[i,:])+1))
  end
  return predic
end

function NonLinearPredictProb(X,X_test,model)
  n = size(X_test,1)
  if model.down == 0
    if model.invK == 0
      model.invK = Matrix(Symmetric(inv(CreateKernelMatrix(X,model.kernelfunction,model.Θ)+model.γ*eye(size(X,1))),:U))
    end
    model.top = model.invK*model.μ
    model.down = -(model.invK+model.invK*model.ζ*model.invK)
  end
  ksize = size(X,1)
  predic = zeros(n)
  k_star = zeros(ksize)
  k_starstar = 0
  for i in 1:n
    for j in 1:ksize
      k_star[j] = model.Kernel_function(X[j,:],X_test[i,:])
    end
    k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])
    predic[i] = cdf(Normal(),(dot(k_star,model.top))/(k_starstar + dot(k_star,model.down*k_star) + 1))
  end
  predic
end

function NonLinearPredict(X,X_test,model)
  n = size(X_test,1)
  if model.top == 0
    model.top = model.invK*model.μ
  end
  k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=X)
  return k_star*model.top
end


function SparsePredictProb(X_test,model)
  n = size(X_test,1)
  ksize = model.m
  if model.down == 0
    if model.top == 0
      model.top = model.invKmm*model.μ
    end
    model.down = model.invKmm*(eye(ksize)+model.ζ*model.invKmm)
    model.MatricesPrecomputed = true
  end
  predic = zeros(n)
  k_star = zeros(ksize)
  k_starstar = 0
  for i in 1:n
    for j in 1:ksize
      k_star[j] = model.Kernel_function(model.inducingPoints[j,:],X_test[i,:])
    end
    k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])

    predic[i] = cdf(Normal(),(dot(k_star,model.top))/(k_starstar - dot(k_star,model.down*k_star) + 1))
  end
  predic
end


function SparsePredict(X_test,model)
  n = size(X_test,1)
  if model.top == 0
    model.top = model.invKmm*model.μ
  end
  k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=model.inducingPoints)
  return k_star*model.top
end

#=function SparsePredictProb(X,X_test,model)
  n = size(X_test,1)
  #if model.down == 0
  #  if model.top == 0
      model.K = CreateKernelMatrix(X,model.Kernel_function)+model.γ*eye(size(X,1))
      model.invK = Matrix(Symmetric(inv(model.K),:U))
      model.top = model.invK*model.κ*model.μ
  #  end
    KtildeComplete = model.K - model.κ*CreateKernelMatrix(model.inducingPoints,model.Kernel_function,X2=X)
    model.down = model.invK + model.invK*(KtildeComplete - model.κ*model.ζ*(model.κ'))
    model.MatricesPrecomputed = true
  #end
  ksize = size(X,1)
  predic = zeros(n)
  k_star = zeros(ksize)
  k_starstar = 0
  for i in 1:n
    for j in 1:ksize
      k_star[j] = model.Kernel_function(X[j,:],X_test[i,:])
    end
    k_starstar = model.Kernel_function(X_test[i,:],X_test[i,:])
    predic[i] = cdf(Normal(),(dot(k_star,model.top))/(k_starstar - dot(k_star,model.down*k_star) + 1))
  end
  predic
end


function SparsePredict(X,X_test,model)
  n = size(X_test,1)
  #if model.top == 0
    model.K = CreateKernelMatrix(X,model.Kernel_function)+model.γ*eye(size(X,1))
    model.invK = Matrix(Symmetric(inv(model.K)))
    model.top = model.invK*model.κ*model.μ
  #end
  k_star = CreateKernelMatrix(X_test,model.Kernel_function,X2=X)
  return k_star*model.top
end=#



    #Update of the noise
    #=γs = linspace(0.01,0.02,200)
    ELBOs = zeros(length(γs))
    grad_ELBO = zeros(length(γs)-1)
    grads = zeros(length(γs))
    for i in 1:length(γs)

      model.K = CreateKernelMatrix(X,model.Kernel_function)+γs[i]*eye(size(X,1))
      model.invK = Matrix(Symmetric(inv(model.K),:U))
      V = model.invK*model.kernels[1].coeff*CreateKernelMatrix(X,model.kernels[1].compute_deriv)
      A = model.invK*model.ζ-eye(model.nFeatures)
      #grads[i]  = 0.5*(trace(V*A)+dot(model.μ,V*model.invK*model.μ))
      ELBOs[i] = model.ELBO(X,y)
      if i%100 == 0
        println(i)
      end
    end
    #println(grads)

    steps =(γs[2:end]-γs[1:end-1])
    grad_ELBO = (ELBOs[2:end]-ELBOs[1:end-1])./steps
    #println(grad_ELBO)
    semilogx(γs,grads,color="blue")
    semilogx(γs,ELBOs,color="red")
    semilogx(γs[1:end-1],grad_ELBO,color="green")

=#
#The traces are replaced by sums of Hadamard product (entrywise) of matrix
