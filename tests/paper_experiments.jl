#### Paper_Experiment_Predictions ####
# Run on a file and compute accuracy on a nFold cross validation
# Compute also the brier score and the logscore

if !isdefined(:DataAccess); include("../src/DataAccess.jl"); end;
if !isdefined(:TestFunctions); include("../src/paper_experiment_functions.jl");end;
using TestFunctions
using PyPlot
using DataAccess
#Compare Platt, B-BSVM, ECM and GPC

#Methods and scores to test
doBBSVM = false
doSBSVM = false
doPlatt = true
doGPC = false
doECM = true

doTime = true #Return time needed for training
doAccuracy = true #Return Accuracy
doBrierScore = true # Return BrierScore
doLogScore = false #Return LogScore

doWrite = false #Write results in approprate folder
ShowIntResults = false #Show intermediate time, and results for each fold
#Testing Parameters
#= Datasets available are get_X :
Ionosphere,Sonar,Crabs,USPS, Banana, Image, RingNorm
BreastCancer, Titanic, Splice, Diabetis, Thyroid, Heart, Waveform, Flare
=#
(X_data,y_data,DatasetName) = get_BreastCancer()
MaxIter = 100 #Maximum number of iterations for every algorithm
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold


#Main Parameters
main_param = DefaultParameters()
main_param["nFeatures"] = nFeatures
main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-5 #Convergence criterium
main_param["M"] = min(100,floor(Int64,0.2*nSamples))
main_param["Kernel"] = "rbf"
main_param["Θ"] = 5.0 #Hyperparameter of the kernel
main_param["BatchSize"] = 10
main_param["Verbose"] = false
main_param["Window"] = 30
#BSVM and GPC Parameters
BBSVMParam = BSVMParameters(Stochastic=false,Sparse=false,ALR=false,main_param=main_param)
SBSVMParam = BSVMParameters(Stochastic=true,Sparse=true,ALR=false,main_param=main_param)
GPCParam = GPCParameters(Stochastic=false,Sparse=false,main_param=main_param)
ECMParam = ECMParameters(main_param=main_param)
SVMParam = SVMParameters(main_param=main_param)

#Global variables for debugging
X = []; y = []; X_test = []; y_test = [];

#Set of all models
TestModels = Dict{String,TestingModel}()

if doBBSVM; TestModels["BBSVM"] = TestingModel("BBSVM",DatasetName,"Prediction","BSVM",BBSVMParam); end;
if doSBSVM; TestModels["SBSVM"] = TestingModel("SBSVM",DatasetName,"Prediction","BSVM",SBSVMParam); end;
if doPlatt; TestModels["Platt"] = TestingModel("SVM",DatasetName,"Prediction","SVM",SVMParam);      end;
if doGPC;   TestModels["GPC"]   = TestingModel("GPC",DatasetName,"Prediction","GPC",GPCParam);      end;
if doECM;   TestModels["ECM"]   = TestingModel("ECM",DatasetName,"Prediction","ECM",ECMParam);      end;

writing_order = Array{String,1}();                    if doTime; push!(writing_order,"time"); end;
if doAccuracy; push!(writing_order,"accuracy"); end;  if doBrierScore; push!(writing_order,"brierscore"); end;
if doLogScore; push!(writing_order,"logscore"); end;

#conv_BSVM = falses(nFold); conv_SBSVM = falses(nFold); conv_SSBSVM = falses(nFold); conv_GPC = falses(nFold); conv_SGPC = falses(nFold); conv_SSGPC = falses(nFold); conv_EM = falses(nFold); conv_FITCEM = falses(nFold); conv_SVM = falses(nFold)
for (name,testmodel) in TestModels
  println("Running $(testmodel.MethodName) on $(testmodel.DatasetName) dataset")
  #Initialize the results storage
  if doTime;        testmodel.Results["time"]       = Array{Float64,1}();end;
  if doAccuracy;    testmodel.Results["accuracy"]   = Array{Float64,1}();end;
  if doBrierScore;  testmodel.Results["brierscore"] = Array{Float64,1}();end;
  if doLogScore;    testmodel.Results["logscore"]   = Array{Float64,1}();end;
  for i in 1:nFold #Run over all folds of the data
    if ShowIntResults
      println("#### Fold number $i/$nFold ###")
    end
    X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
    y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
    X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
    y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
    CreateModel(testmodel,X,y)
    time = TrainModel(testmodel,X,y,MaxIter)
    if ShowIntResults
      println("$(testmodel.MethodName) : Time  = $time")
    end
    if doTime; push!(testmodel.Results["time"],time); end;
    RunTests(testmodel,X,X_test,y_test,accuracy=doAccuracy,brierscore=doBrierScore,logscore=doLogScore)
  end
  ProcessResults(testmodel,writing_order) #Compute mean and std deviation
  PrintResults(testmodel.Results["allresults"],testmodel.MethodName,writing_order) #Print the Results in the end
  if doWrite
    top_fold = "data";
    if !isdir(top_fold); mkdir(top_fold); end;
    WriteResults(testmodel,top_fold) #Write the results in an adapted format into a folder
  end
end
