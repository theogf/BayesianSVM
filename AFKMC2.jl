#File for the Assumption Free K MC2 algorithm
using Distributions
using StatsBase
using Clustering


function KMeansInducingPoints(X,K,m)
  C = (KmeansSeed(X,K,m))'
  kmeans!(X',C)
  return C'
end

function KmeansSeed(X,K,m) #X is the data, K the number of centers wanted, m the number of Markov iterations
  N = size(X,1)
  #Preprocessing, sample first random center
  init = StatsBase.sample(1:N,1)
  C = zeros(K,size(X,2))
  C[1,:] = X[init,:]
  q = zeros(N)
  for i in 1:N
    q[i] = 0.5*norm(X[i,:]-C[1])^2
  end
  sumq = sum(q)
  q = WeightVec(q/sumq + 1.0/(2*N),1)
  uniform = Distributions.Uniform(0,1)
  for i in 2:K
    x = X[StatsBase.sample(1:N,q,1),:] #weighted sampling,
    mindist = mindistance(x,C,i-1)
    for j in 2:m
      y = X[StatsBase.sample(q),:] #weighted sampling
      dist = mindistance(y,C,i-1)
      if (dist/mindist > rand(uniform))
        x = y;  mindist = dist
      end
    end
    C[i,:]=x
  end
  return C
end

function mindistance(x,C,K) #Point to look for, collection of centers, number of centers computed
  mindist = Inf
  for i in 1:K
    mindist = min(norm(x-C[i])^2,mindist)
  end
  return mindist
end

#(X_data,y_data) = generate_two_multivariate_data(2,2000; sep_ = 2.0, y_neg_= true)
#=(X_data,y_data) = get_USPS()
time = @elapsed C = Kmeans(X_data,50,10)
println("took $time s")
dim1 = 5; dim2 = 6;
plot(X_data[:,dim1],X_data[:,dim2],linestyle="None",marker="o",color="red")
plot(C[:,dim1],C[:,dim2],linestyle="None",marker="o",color="blue",markersize=15.0)
C=C'
time = @elapsed kmeans!(X_data',C)
println("took $time s")
C=C'
plot(C[:,dim1],C[:,dim2],linestyle="None",marker="o",color="green",markersize=10.0)
=#
