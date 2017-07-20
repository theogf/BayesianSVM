#### Paper_Experiment_Predictions ####
# Run on a file and compute accuracy on a nFold cross validation
# Compute also the brier score and the logscore

if !isdefined(:DataAccess); include("../src/DataAccess.jl"); end;
if !isdefined(:TestFunctions); include("../src/test_functions.jl");end;
using TestFunctions
using DataAccess

####### Data and Training Parameters #######
### Setting the Dataset ###
#= Datasets available with get_X , replace X with :
Ionosphere,Sonar,Crabs,USPS, Banana, Image, RingNorm
BreastCancer, Titanic, Splice, Diabetis, Thyroid, Heart, Waveform, Flare (a package will be soon created to deal with user's datasets)
=#
(X_data,y_data,DatasetName) = get_BreastCancer()
(nSamples,nFeatures) = size(X_data);
nFold = 10; #Chose the number of folds
fold_separation = collect(1:nSamples÷nFold:nSamples+1) #Separate the data in nFold

### Setting results output ###
doTime = true #Compute time needed for training
doAccuracy = true #Compute Accuracy
doBrierScore = true #Compute BrierScore
doLogScore = false #Compute LogScore
doWrite = false #Write results in appropriate folder
ShowIntResults = false #Show intermediate time, and results for each fold


### Training Parameters and Hyperparameters###
MaxIter = 100 #Maximum number of iterations for every algorithm
main_param = DefaultParameters(); main_param["nFeatures"] = nFeatures; main_param["nSamples"] = nSamples
main_param["ϵ"] = 1e-5 #Convergence criterium;
main_param["M"] = min(100,floor(Int64,0.2*nSamples));main_param["Kernel"] = "rbf"
main_param["Θ"] = 5.0 #Hyperparameter of the kernel
main_param["BatchSize"] = 10;main_param["Verbose"] = false;main_param["Window"] = 30

#### Creating the Model ####
#BSVM Parameters (Stochastic is for the Stochastic version, Sparse is with induing points and ALR is adaptative learning rate)
SBSVMParam = BSVMParameters(Stochastic=true,Sparse=true,ALR=false,main_param=main_param)
Model = TestingModel("SBSVM",DatasetName,"Prediction","BSVM",SBSVMParam)

writing_order = Array{String,1}();                    if doTime; push!(writing_order,"time"); end;
if doAccuracy; push!(writing_order,"accuracy"); end;  if doBrierScore; push!(writing_order,"brierscore"); end;
if doLogScore; push!(writing_order,"logscore"); end;
#Initialize the results storage
if doTime;        Model.Results["time"]       = Array{Float64,1}();end; if doAccuracy;    Model.Results["accuracy"]   = Array{Float64,1}();end;
if doBrierScore;  Model.Results["brierscore"] = Array{Float64,1}();end; if doLogScore;    Model.Results["logscore"]   = Array{Float64,1}();end;
for i in 1:nFold #Run over all folds of the data


###### Training  Model ######
println("Running SBVM on $(Model.DatasetName) dataset")
  if ShowIntResults
    println("#### Fold number $i/$nFold ###")
  end
  #Separating Data
  X_test = X_data[fold_separation[i]:(fold_separation[i+1])-1,:]
  y_test = y_data[fold_separation[i]:(fold_separation[i+1])-1]
  X = X_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples)),:]
  y = y_data[vcat(collect(1:fold_separation[i]-1),collect(fold_separation[i+1]:nSamples))]
  #Add Data to the Model
  CreateModel(Model,X,y)
  #Train the Model
  time = TrainModel(Model,X,y,MaxIter)
  if ShowIntResults
    println("$(Model.MethodName) : Time  = $time")
  end
  if doTime; push!(Model.Results["time"],time); end;
  #Run the tests on the last fold of data
  RunTests(Model,X,X_test,y_test,accuracy=doAccuracy,brierscore=doBrierScore,logscore=doLogScore)
end

##### Process the results from the k-fold ####
ProcessResults(Model,writing_order) #Compute mean and std deviation
PrintResults(Model.Results["allresults"],testmodel.MethodName,writing_order) #Print the Results in the end
if doWrite
  top_fold = "data";
  if !isdir(top_fold); mkdir(top_fold); end;
  WriteResults(Model,top_fold) #Write the results in an adapted format into a folder
end
