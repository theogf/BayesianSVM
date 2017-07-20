# Paper_Experiment_Functions.jl
#= ---------------- #
Set of datatype and functions for efficient testing.
# ---------------- =#

include("BSVM.jl")

module TestFunctions

using ScikitLearn
using Distributions
using KernelFunctions

export TestingModel
export DefaultParameters, BSVMParameters
export CreateModel, TrainModel, RunTests, ProcessResults, PrintResults, WriteResults
export ComputePrediction, ComputePredictionAccuracy

#Datatype for containing the model, its results and its parameters
type TestingModel
  MethodName::String #Name of the method
  DatasetName::String #Name of the dataset
  ExperimentType::String #Type of experiment
  MethodType::String #Type of method used ("SVM","BSVM","ECM","GPC")
  Param::Dict{String,Any} #Some paramters to run the method
  Results::Dict{String,Any} #Saved results
  Model::Any
  TestingModel(methname,dataset,exp,methtype) = new(methname,dataset,exp,methtype,Dict{String,Any}(),Dict{String,Any}())
  TestingModel(methname,dataset,exp,methtype,params) = new(methname,dataset,exp,methtype,params,Dict{String,Any}())
end

#Create a default dictionary
function DefaultParameters()
  param = Dict{String,Any}()
  param["ϵ"]= 1e-8 #Convergence criteria
  param["BatchSize"] = 10 #Number of points used for stochasticity
  param["Kernel"] = "rbf" # Kernel function
  param["Θ"] = 1.0 # Hyperparameter for the kernel function
  param["γ"] = 1.0 #Variance of introduced noise
  param["M"] = 32 #Number of inducing points
  param["Window"] = 5 #Number of points used to check convergence (smoothing effect)
  param["Verbose"] = 0 #Verbose
  return param
end

#Create a default parameters dictionary for BSVM
function BSVMParameters(;Stochastic=true,NonLinear=true,Sparse=true,ALR=true,Autotuning=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["Stochastic"] = Stochastic #Is the method stochastic
  param["Sparse"] = Sparse #Is the method using inducing points
  param["NonLinear"] = NonLinear #Is the method using kernels
  param["ALR"] = ALR #Is the method using adpative learning rate (in case of the stochastic case)
  param["AutoTuning"] = Autotuning #Is hyperoptimization performed
  param["ATFrequency"] = 10 #How even autotuning is performed
  param["κ"] = 1.0;  param["τ"] = 100; #Parameters for learning rate of autotuning
  param["κ_s"] = 1.0;  param["τ_s"] = 100; #Parameters for learning rate of Stochastic gradient descent when ALR is not used
  param["ϵ"] = main_param["ϵ"]; param["Window"] = main_param["Window"]; #Convergence criteria (checking parameters norm variation on a window)
  param["Kernels"] = [Kernel(main_param["Kernel"],1.0,params=main_param["Θ"])] #Kernel creation (standardized for now)
  param["Verbose"] = main_param["Verbose"] ? 2 : 0 #Verbose
  param["BatchSize"] = main_param["BatchSize"] #Number of points used for stochasticity
  param["M"] = main_param["M"] #Number of inducing points
  param["γ"] = main_param["γ"] #Variance of introduced noise
  return param
end

#Create a default parameters dictionary for GPC (similar to BSVM)
function GPCParameters(;Sparse=true,Stochastic=false,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["Sparse"] = Sparse
  if Sparse
    param["Stochastic"] = Stochastic
  else
    param["Stochastic"] = false
  end
  param["ϵ"] = main_param["ϵ"]
  param["Kernel"] = GPflow.kernels[:Add]([GPflow.kernels[:RBF](main_param["nFeatures"]),GPflow.kernels[:White](input_dim=main_param["nFeatures"],variance=main_param["γ"])])
  param["BatchSize"] = main_param["BatchSize"]
  param["M"] = main_param["M"]
  return param
end

#Create a default parameters  dictionary for ECM (similar to BSVM)
function ECMParameters(;main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["ϵ"] = main_param["ϵ"]
  param["Kernel"] = Kernel(main_param["Kernel"],1.0,params=main_param["Θ"])
  param["γ"] = main_param["γ"]
  param["Verbose"] = main_param["Verbose"]
  return param
end


#Create a default parameters  dictionary for SVM (similar to BSVM)
function SVMParameters(;probability = true,main_param=DefaultParameters())
  param = Dict{String,Any}()
  param["probability"] = probability
  param["ϵ"] = main_param["ϵ"]
  param["Kernel"] = main_param["Kernel"]
  param["gamma"] = 1/(main_param["Θ"]^2)
  param["C"] = 2*main_param["γ"]
  param["Verbose"] = main_param["Verbose"]
  return param
end

#Create a model given the parameters passed in p
function CreateModel(tm::TestingModel,X,y) #tm testing_model, p parameters
  if tm.MethodType == "BSVM"
    tm.Model = BSVM(tm.Param["Stochastic"],batchSize=tm.Param["BatchSize"],Sparse=tm.Param["Sparse"],m=tm.Param["M"],NonLinear=tm.Param["NonLinear"],
    kernels=tm.Param["Kernels"],Autotuning=tm.Param["AutoTuning"],autotuningfrequency=tm.Param["ATFrequency"],AdaptativeLearningRate=tm.Param["ALR"],κ_s=tm.Param["κ_s"],τ_s = tm.Param["τ_s"],ϵ=tm.Param["ϵ"],γ=tm.Param["γ"],
    κ_Θ=tm.Param["κ"],τ_Θ=tm.Param["τ"],smoothingWindow=tm.Param["Window"],VerboseLevel=tm.Param["Verbose"])
  elseif tm.MethodType == "GPC"
    if tm.Param["Sparse"]
      if tm.Param["Stochastic"]
        #Stochastic Sparse GPC model
        GPflow.svgp[:SVGP](X, reshape((y+1)./2,(length(y),1)),kern=tm.Param["Kernel"], likelihood=GPflow.likelihoods[:Bernoulli](), Z=KMeansInducingPoints(X,tm.Param["M"],10), minibatch_size=tm.Param["BatchSize"])
      else
        #Sparse GPC model
        GPflow.svgp[:SVGP](X, reshape((y+1)./2,(size(y,1),1)),kern=tm.Param["Kernel"],likelihood=GPflow.likelihoods[:Bernoulli](), Z=KMeansInducingPoints(X,tm.Param["M"],10))
      end
    else
      #Basic GPC model
      tm.Model = GPflow.vgp[:VGP](X, reshape((y+1)./2,(size(y,1),1)),kern=tm.Param["Kernel"],likelihood=GPflow.likelihoods[:Bernoulli]())
    end
  elseif tm.MethodType == "SVM"
    tm.Model = SVC(C=tm.Param["C"],gamma=tm.Param["gamma"], kernel=tm.Param["Kernel"],probability=tm.Param["probability"],tol=tm.Param["ϵ"],verbose=tm.Param["Verbose"])
  end
end

#Train the model on trainin set (X,y) for #iterations
function TrainModel(tm::TestingModel,X,y,iterations)
  time_training = 0;
  if tm.MethodType == "BSVM"
    tm.Model.nEpochs = iterations
    time_training = @elapsed TrainBSVM(tm.Model,X,y)
  elseif tm.MethodType == "GPC"
    time_training = @elapsed tm.Model[:optimize](maxiter=iterations)
  elseif tm.MethodType == "SVM"
    tm.Model[:max_iter] = iterations
    time_training = @elapsed tm.Model[:fit](X,y)
  elseif tm.MethodType == "ECM"
    time_training = @elapsed tm.Model = ECMTraining(X,y,γ=tm.Param["γ"],nepochs=iterations,ϵ=tm.Param["ϵ"],kernel=tm.Param["Kernel"],verbose=tm.Param["Verbose"])
  end
  return time_training;
end

#Run tests accordingly to the arguments and save them
function RunTests(tm::TestingModel,X,X_test,y_test;accuracy::Bool=false,brierscore::Bool=false,logscore::Bool=false)
  if accuracy
    push!(tm.Results["accuracy"],TestAccuracy(y_test,ComputePrediction(tm,X,X_test)))
  end
  y_predic_acc = 0
  if brierscore
    y_predic_acc = ComputePredictionAccuracy(tm::TestingModel, X, X_test)
    push!(tm.Results["brierscore"],TestBrierScore(y_test,y_predic_acc))
  end
  if logscore
    if y_predic_acc == 0
      y_predic_acc = ComputePredictionAccuracy(tm::TestingModel, X, X_test)
    end
    push!(tm.Results["logscore"],TestLogScore(y_test,y_predic_acc))
  end
end


#Compute the mean and the standard deviation and assemble in one result
function ProcessResults(tm::TestingModel,writing_order)
  all_results = Array{Float64,1}()
  names = Array{String,1}()
  for name in writing_order
    result = [mean(tm.Results[name]), std(tm.Results[name])]
    all_results = vcat(all_results,result)
    names = vcat(names,name)
  end
  if haskey(tm.Results,"allresults")
    tm.Results["allresults"] = vcat(tm.Results["allresults"],all_results')
  else
    tm.Results["allresults"] = all_results'
  end
  if !haskey(tm.Results,"names")
    tm.Results["names"] = names
  end
end

function PrintResults(results,method_name,writing_order)
  println("Model $(method_name) : ")
  i = 1
  for category in writing_order
    println("$category : $(results[i*2-1]) ± $(results[i*2])")
    i+=1
  end
end

function WriteResults(tm::TestingModel,location)
  fold = String(location*"/"*tm.ExperimentType*"Experiment_"*tm.DatasetName*"Dataset")
  if !isdir(fold); mkdir(fold); end;
  writedlm(String(fold*"/Results_"*tm.MethodName*".txt"),tm.Results["allresults"])
end

#Return predicted labels (-1,1) for test set X_test
function ComputePrediction(tm::TestingModel, X, X_test)
  y_predic = []
  if tm.MethodType == "BSVM"
    if tm.Model.NonLinear
      y_predic = sign(tm.Model.Predict(X,X_test))
    else
      y_predic = sign(tm.Model.Predict(X_test))
    end
  elseif tm.MethodType == "GPC"
    y_predic = sign(tm.Model[:predict_y](X_test)[1]*2-1)
  elseif tm.MethodType == "SVM"
    y_predic = sign(tm.Model[:predict](X_test))
  elseif tm.MethodType == "ECM"
    y_predic = sign(PredicECM(X,tm.Model[4],X_test,tm.Model[1],tm.Model[2],tm.Param["γ"],tm.Model[3]))
  end
  return y_predic
end

#Return prediction certainty for class 1 on test set X_test
function ComputePredictionAccuracy(tm::TestingModel, X, X_test)
  y_predic = []
  if tm.MethodType == "BSVM"
    if tm.Model.NonLinear
      y_predic = tm.Model.PredictProba(X,X_test)
    else
      y_predic = tm.Model.PredictProba(X_test)
    end
  elseif tm.MethodType == "GPC"
    y_predic = tm.Model[:predict_y](X_test)[1]
  elseif tm.MethodType == "SVM"
    y_predic = tm.Model[:predict_proba](X_test)[:,2]
  elseif tm.MethodType == "ECM"
    y_predic = PredictProbaECM(X,tm.Model[4],X_test,tm.Model[1],tm.Model[2],tm.Param["γ"],tm.Model[3])
  end
  return y_predic
end

#Return Accuracy on test set
function TestAccuracy(y_test, y_predic)
  return 1-sum(1-y_test.*y_predic)/(2*length(y_test))
end
#Return Brier Score
function TestBrierScore(y_test, y_predic)
  return sum(((y_test+1)./2 - y_predic).^2)/length(y_test)
end
#Return Log Score
function TestLogScore(y_test, y_predic)
  return sum((y_test+1)./2.*log(y_predic)+(1-(y_test+1)./2).*log(1-y_predic))/length(y_test)
end
#Return ROC
function TestROC(y_test,y_predic)
    nt = length(y_test)
    truepositive = zeros(npoints); falsepositive = zeros(npoints)
    truenegative = zeros(npoints); falsenegative = zeros(npoints)
    thresh = collect(linspace(0,1,npoints))
    for i in 1:npoints
      for j in 1:nt
        truepositive[i] += (yp[j]>=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
        truenegative[i] += (yp[j]<=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsepositive[i] += (yp[j]>=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsenegative[i] += (yp[j]<=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
      end
    end
    return (truepositive./(truepositive+falsenegative),falsepositive./(truenegative+falsepositive))
end

end #end of module
