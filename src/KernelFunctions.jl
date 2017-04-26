#Module for the kernel type
#

module KernelFunctions

export Kernel
export CreateKernelMatrix, CreateDiagonalKernelMatrix
export delta_kroenecker


type Kernel
    kernel_function::Function # Kernel function
    coeff::Float64 #Weight for the kernel
    derivative_kernel::Function # Derivative of the kernel function (used for hyperparameter optimization)
    param::Float64 #Hyperparameter for the kernel function (depends of the function)
    Nparams::Int64 #Number of hyperparameters
    compute::Function #General computational function
    compute_deriv::Function #General derivative function
    #Constructor
    function Kernel(kernel, coeff::Float64; params=0)
      this = new()
      this.coeff = coeff
      this.Nparams = 1
      if kernel=="rbf"
        this.kernel_function = rbf
        this.derivative_kernel = deriv_rbf
        this.param = params
      elseif kernel=="quadra"
        this.kernel_function = quadratic
        this.derivative_kernel = deriv_quadratic
        this.param = params
      elseif kernel=="linear"
        this.kernel_function = linear
        this.Nparams = 0
      elseif kernel=="laplace"
        this.kernel_function = laplace
        this.derivative_kernel = deriv_laplace
        this.param = params
      elseif kernel=="abel"
        this.kernel_function = abel
        this.derivative_kernel = deriv_abel
        this.param = params
      elseif kernel=="imk"
        this.kernel_function = imk
        this.derivative_kernel = deriv_imk
        this.param = params
      else
        error("Kernel function $(kernel_list[i]) not available, options are : rbf, linear, laplace, abel, imk")
      end
      if this.Nparams > 0
        this.compute = function(X1,X2)
            this.kernel_function(X1,X2,this.param)
          end
        this.compute_deriv = function(X1,X2)
            this.derivative_kernel(X1,X2,this.param)
          end
      else
        this.compute = function(X1,X2)
            this.kernel_function(X1,X2)
          end
      end
      return this
    end
end

function CreateKernelMatrix(X1,kernel_function;X2=0) #Create the kernel matrix from the training data or the correlation matrix between two set of data
  if X2 == 0
    ksize = size(X1,1)
    K = zeros(ksize,ksize)
    for i in 1:ksize
      for j in 1:i
        K[i,j] = kernel_function(X1[i,:],X1[j,:])
        if i != j
          K[j,i] = K[i,j]
        end
      end
    end
    return K
  else
    ksize1 = size(X1,1)
    ksize2 = size(X2,1)
    K = zeros(ksize1,ksize2)
    for i in 1:ksize1
      for j in 1:ksize2
        K[i,j] = kernel_function(X1[i,:],X2[j,:])
      end
    end
    return K
  end
end


function CreateDiagonalKernelMatrix(X,kernel_function;MatrixFormat=false)
  n = size(X,1)
  kermatrix = zeros(n)
  for i in 1:n
    kermatrix[i] = kernel_function(X[i,:],X[i,:])
  end
  if MatrixFormat
    return diagm(kermatrix)
  else
    return kermatrix
  end
end

function delta_kroenecker(X1::Array{Float64,1},X2::Array{Float64,1})
  return X1==X2 ? 1 : 0
end

#Gaussian (RBF) Kernel
function rbf(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  if X1 == X2
    return 1
  end
  exp(-(norm(X1-X2))^2/(Θ^2))
end


function deriv_rbf(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  a = norm(X1-X2)
  if a != 0
    return 2*a^2/(Θ^3)*exp(-a^2/(Θ^2))
  else
    return 0
  end
end

#Laplace Kernel
function laplace(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  if X1 == X2
    return 1
  end
  exp(-Θ*norm(X1-X2,2))
end

function deriv_laplace(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  a = norm(X1-X2,2)
  -a*exp(-Θ*a)
end

#Abel Kernel
function abel(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  if X1==X2
    return 1
  end
  exp(-Θ*norm(X1-X2,1))
end

function deriv_abel(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  a = norm(X1-X2,1)
  -a*exp(-Θ*a)
end

function imk(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  1/(sqrt(norm(X1-X2)^2+Θ))
end

function deriv_imk(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  -1/(2*(sqrt(norm(X1-X2)+Θ))^3)
end

#Linear Kernel
function linear(X1::Array{Float64,1},X2::Array{Float64,1})
  dot(X1,X2)
end

#Quadratic Kernel (special case of polynomial kernel)
function quadratic(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  (dot(X1,X2)+Θ)^2
end

function deriv_quadratic(X1::Array{Float64,1},X2::Array{Float64,1},Θ)
  2*(dot(X1,X2)+Θ)
end

end #end of module
