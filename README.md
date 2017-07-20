## README ##


### Objective ###

* This repository contains the updated source code for the ***Bayesian Nonlinear Support Vector Machine (BSVM)*** both in its **stochastic (and with inducing points)** and its **batch version**
* It relates to the paper submitted to ECML 17' __"Bayesian Nonlinear Support Vector Machines for Big Data"__ by Florian Wenzel, Theo Galy-Fajou, Matthäus Deutsch and Marius Kloft. Paper is available at [https://arxiv.org/abs/1707.05532][arxiv]

### How do I install the package? ###

* First clone this repository (`git clone https://github.com/theogf/BayesianSVM.git`)
* If you simply want to try out the package you need to install the **Julia** dependencies :
    - [Distributions][dist]
    - [PyPlot][pyplot]
    - [StatsBase][statsbase]
    - [GaussianMixtures][gaussm]
    - [Clustering][clustering]
    - [Scikitlearn][scikitjl]
    
    *Note: to install new packages use the Pkg.add("ModuleName") function in Julia*
* If you want to try the competitors as well you will need to install these **Julia** and **Python** dependencies (as well as Python ofc): 
    * (Julia)[PyCall][pycall]
    * (Python)[Scikitlearn][scikit]
    * (Python)[Tensorflow][tflow]
    * (Python)[GPflow][gpflow]
    
    *Note: to use Tensorflow and GPflow, they must me included in the search path of PyCall, to do this use : `unshift!(PyVector(pyimport("sys")["path"]), "path_to_add")` and call `Pkg.build("PyCall")`, also note that they are much more complicate to install*
* Both tests and source files are written in Julia (v0.5), one first needs to julia to run those, however a Python or Matlab user should be able to read easily through the code as the syntax is quite similar
* Some light datasets are included (especially the **Rätsch Benchmark dataset**), the SUSY dataset can be found on UCI
### How to run tests? ###

* Go to the "test" folder, open "run_test.jl", chose the dataset and change the parameters (more is explained in the file) and simply run the file. (*for example change the type of BSVM (linear/nonlinear, sparse, use of stochasticity etc*)
* If you want to also use the competitors, open "paper_experiments.jl", chose the dataset, chose the methods you want to test and adapt the parameters (more details in the file).
* For more custom usage of the BSVM method, look at the source code of src/BSVM.jl, where all the options are explained. More documentation will be there soon.

### Who to contact ###

**For any queries please contact theo.galyfajou at gmail.com**

   [arxiv]: <https://arxiv.org/abs/1707.05532>
   [dist]: <https://github.com/JuliaStats/Distributions.jl>
   [pyplot]: <https://github.com/JuliaPy/PyPlot.jl>
   [pycall]:<https://github.com/JuliaPy/PyCall.jl>
   [statsbase]:<https://github.com/JuliaStats/StatsBase.jl>
   [gaussm]:<https://github.com/davidavdav/GaussianMixtures.jl>
   [clustering]:<https://github.com/JuliaStats/Clustering.jl>
   [scikitjl]:<https://github.com/cstjean/ScikitLearn.jl>
   [scikit]:<http://scikit-learn.org/stable/>
   [tflow]:<https://www.tensorflow.org/>
   [gpflow]:<https://github.com/GPflow/GPflow>
