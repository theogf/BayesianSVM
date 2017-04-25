#Module for either generating data or exporting from an existing dataset

module DataAccess

using Distributions

export generate_uniform_data, generate_normal_data, generate_two_multivariate_data
export get_Ionosphere, get_Sonar, get_Crabs, get_USPS
export get_SUSY, get_Banana, get_Image, get_RingNorm
export get_BinaryMNIST, get_3vs5MNIST, get_BreastCancer
export get_Titanic, get_Splice, get_Diabetis, get_Thyroid
export get_Heart, get_Waveform, get_Flare


function generate_uniform_data(nFeatures_::Int64,nSamples_::Int64; y_neg_::Bool = true, boxsize_::Float64 = 1.0, noise_::Float64 = 0.3)
  generate_data(2, nFeatures_, nSamples_, y_neg = y_neg_, range = boxsize_, noise = noise_)
end

function generate_two_multivariate_data(nFeatures_::Int64,nSamples_::Int64; sep_::Float64 = 2.0, y_neg_::Bool = true, noise_::Float64 = 1.0)
  generate_data(1, nFeatures_, nSamples_, y_neg = y_neg_, noise = noise_)
end

function generate_normal_data(nFeatures_::Int64,nSamples_::Int64; sep_::Float64 = 2.0, y_neg_::Bool = true,noise_::Float64 = 0.3, σ_::Float64 = 1.0)
  generate_data(3, nFeatures_, nSamples_, y_neg = y_neg_, noise = noise_, σ=σ_)
end

function generate_multi_beta_normal_data(nFeatures_::Int64,nSamples_::Int64; sep_::Float64 = 2.0, y_neg_::Bool = true,noise_::Float64 = 0.3, σ_::Float64 = 1.0,nβ_::Int64 = 4)
  generate_data(4, nFeatures_, nSamples_, y_neg = y_neg_, noise = noise_, σ=σ_, nβ=nβ_)
end

function get_Ionosphere()
  generate_data(5,0,0)
end

function get_Sonar()
  generate_data(6,0,0)
end

function get_Crabs()
  generate_data(7,0,0)
end

function get_Pima()
  generate_data(8,0,0)
end

function get_USPS()
  generate_data(9,0,0)
end

function get_SUSY()
  generate_data(10,0,0)
end

function get_Banana()
  generate_data(11,0,0)
end

function get_Image()
  generate_data(12,0,0)
end

function get_RingNorm()
  generate_data(17,0,0)
end

function get_BinaryMNIST()
  generate_data(22,0,0)
end

function get_3vs5MNIST()
  generate_data(23,0,0)
end


function get_BreastCancer()
  generate_data(25,0,0)
end

function get_Titanic()
  generate_data(26,0,0)
end

function get_Splice()
  generate_data(27,0,0)
end

function get_Diabetis()
  generate_data(28,0,0)
end

function get_Thyroid()
  generate_data(30,0,0)
end

function get_Heart()
  generate_data(31,0,0)
end

function get_German()
  generate_data(32,0,0)
end
function get_Waveform()
  generate_data(33,0,0)
end

function get_Flare()
  generate_data(34,0,0)
end

function generate_data(datatype, nFeatures, nSamples; sep = 2, y_neg::Bool = true, range::Float64 = 1.0, noise::Float64=1.0, σ::Float64=1.0, nβ::Int64=4)
    β_true = zeros(nFeatures)
    accuracy = 1
    shuffling = true
    seed = 123
    DatasetName = "None"
    if datatype == 1 # Multivariate normal distributed mixture X, easy separable
        X = randn(MersenneTwister(seed),(nSamples,nFeatures))
        X[1:nSamples÷2,:] += sep
        X[nSamples÷2+1:end,:] -= sep
        y = zeros((nSamples,1))
        y[1:nSamples÷2] += 1
        β_true[:] = sep
    elseif datatype == 2 #Generate points in a box (range) uniformly and separate them through a true hyperplane, still allowing mistakes with a normal (noise)
        X = rand(Uniform(-range,range),(nSamples,nFeatures))
        β_true = rand(Normal(0,1),nFeatures)
        y = sign(X*β_true+rand(Normal(0,noise),nSamples))
        accuracy= 1-countnz(y-sign(X*β_true))/nSamples
        y_neg = false
    elseif datatype == 3
        X = rand(IsoNormal(zeros(nFeatures),PDMats.ScalMat(nFeatures,σ)),nSamples)'
        β_true = rand(Normal(0,1),nFeatures)
        y = sign(X*β_true+rand(Normal(0,noise),nSamples))
        accuracy= 1-countnz(y-sign(X*β_true))/nSamples
        y_neg = false
    elseif datatype == 4
        X = rand(IsoNormal(zeros(nFeatures),PDMats.ScalMat(nFeatures,σ)),nSamples)'
        β_true = rand(Normal(0,1),(nFeatures,nβ))
        y = zeros((nSamples,1))
        for i in 1:nSamples
          y[i] = sign(X[i,:]*β_true[(nSamples%nβ)+1]+rand(Normal(0,noise),nSamples))
        end
        β_true = mean(β_true)
        accuracy = 1-countnz(y-sign(X*β_true))/nSamples
        y_neg = false
    elseif datatype == 5
        data = readdlm("../data/ionosphere.data",',')
        X = convert(Array{Float64,2},data[:,1:(end-1)])
        y = convert(Array{Float64,1},collect(data[:,end].=="g"))
        nSamples = size(X,1)
        shuffling = true
        DatasetName = "Ionosphere"
    elseif datatype == 6
        data = readdlm("../data/sonar.data",',')
        X = convert(Array{Float64,2},data[:,1:(end-1)])
        y = convert(Array{Float64,1},collect(data[:,end].=="M"))
        nSamples = size(X,1)
        shuffling = true
        DatasetName = "Sonar"
    elseif datatype == 7
        data = readdlm("../data/crabs.csv",',')
        X = data[2:end,3:end]
        X[:,1] = convert(Array{Float64,1},collect(X[:,1].=="M"))
        X = convert(Array{Float64,2},X)
        y = convert(Array{Float64,1},collect(data[2:end,2].=="B"))
        nSamples = size(X,1)
        shuffling = true
        DatasetName = "Crabs"
    elseif datatype == 8
        data = readdlm("../data/pima-indians-diabetes.data",',')
        X = convert(Array{Float64,2},data[:,1:(end-1)])
        y = convert(Array{Float64,1},data[:,end])
        nSamples = size(X,1)
        shuffling = true
        DatasetName = "Pima"
    elseif datatype == 9
        data = readdlm("../data/USPS.csv",';')
        X = convert(Array{Float64,2},data[:,2:end])
        y = convert(Array{Float64,1},collect(data[:,1].==3))
        nSamples = size(X,1)
        shuffling = true
        DatasetName = "USPS"
    elseif datatype == 10
        data = readdlm("../data/Processed_SUSY.data",',')
        X = convert(Array{Float64,2},data[:,1:end-1])
        y = convert(Array{Float64,1},data[:,end])
        nSamples = size(X,1)
        shuffling = false
        DatasetName = "SUSY"
    elseif datatype == 11
        data = readdlm("../data/banana_data.csv",',')
        X = convert(Array{Float64,2},data[:,2:end])
        y = convert(Array{Float64,1},data[:,1])
        nSamples = size(X,1)
        DatasetName = "Banana"
        y_neg = false
    elseif datatype == 12
        data = readdlm("../data/Processed_Image.data",',')
        X = convert(Array{Float64,2},data[:,1:end-1])
        y = convert(Array{Float64,1},data[:,end])
        nSamples = size(X,1)
        DatasetName = "Image"
        shuffling = false
        y_neg = false
    elseif datatype == 17
        data = readdlm("../data/ringnorm.data")
        X = convert(Array{Float64,2},data[:,1:end-1])
        y = convert(Array{Float64,1},data[:,end])
        nSamples = size(X,1)
        shuffling = false
        DatasetName = "RingNorm"
    elseif datatype == 22
      data = readdlm("../data/Processed_BinaryMNIST.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "BinaryMNIST"
    elseif datatype == 23
      data = readdlm("../data/Processed_3vs5MNIST.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "3vs5MNIST"
    elseif datatype == 25
      data = readdlm("../data/Processed_BreastCancer.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "BreastCancer"
    elseif datatype == 26
      data = readdlm("../data/Processed_Titanic.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "Titanic"
    elseif datatype == 27
      data = readdlm("../data/Processed_Splice.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "Splice"
    elseif datatype == 28
      data = readdlm("../data/Processed_Diabetis.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "Diabetis"
    elseif datatype == 30
      data = readdlm("../data/Processed_Thyroid.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "Thyroid"
    elseif datatype == 31
      data = readdlm("../data/Processed_Heart.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "Heart"
    elseif datatype == 32
      data = readdlm("../data/Processed_German.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "German"
    elseif datatype == 33
      data = readdlm("../data/Processed_Waveform.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "Waveform"
    elseif datatype == 34
      data = readdlm("../data/Processed_Flare.data",',')
      X = convert(Array{Float64,2},data[:,1:end-1])
      y = convert(Array{Float64,1},data[:,end])
      nSamples = size(X,1)
      shuffling = false
      y_neg = false
      DatasetName = "Flare"
    end

    #Go from y [0,1] to y [-1,1]
    if y_neg
         y = 2*y-1
    end
    Z = hcat(X,y)
    if shuffling
      Z = Z[shuffle(collect(1:nSamples)),:] #Shuffle the data
    end
    (Z[:,1:end-1],Z[:,end],DatasetName,β_true,accuracy) #Send the data and the parameters separately
end;

end #end of module
