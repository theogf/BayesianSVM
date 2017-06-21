if !isdefined(:DataAccess); include("../src/DataAccess.jl"); end;
if !isdefined(:KernelFunctions); include("KernelFunctions.jl"); end;
using KernelFunctions;
using DataAccess;
using Plotly;

function toeplitz{T}(x::Array{T})
    n = length(x)
    A = zeros(T, n, n)
    for i in 1:n
        A[i,i:end] = x[1:n - i + 1]
        A[i:end,i] = x[1:n - i + 1]
    end
    return A
end

function cubics_coeff(x)
  y = zeros(size(x)); x = abs(x);
  q = find(x.<=1);          # Coefficients:  1.5, -2.5,  0, 1
  y[q] =            (( 1.5 * x[q] - 2.5) .* x[q]    ) .* x[q] + 1;
  q = find(1.<x .== x.<=2);    # Coefficients: -0.5,  2.5, -4, 2
  y[q] =            ((-0.5 * x[q] + 2.5) .* x[q] - 4) .* x[q] + 2;
  return y
end


dim = 2; γ = 0.0001; n = 1000;
(X,y) = generate_normal_data(dim,n; noise_   = 1.0,σ_=25.0)
X= reshape(sort(X[:]),(n,dim))
kern = Kernel("rbf",1.0,params= 10.0)
kernels = Kernel_Combination([kern]);
npoints = 40
# firsttry = @elapsed invK = getInverseKISSMatrix(X,k,γ,npoints =10)
function getKISSMatrices(X,kernels::Kernel_Combination,γ; npoints = 100)
  #Compute maxima and minima
  (nx,d) = size(X);
  cubic = true;
  #indices for the spline interpolation
  if cubic
    its = collect(-2:1:1)
  else
    its = collect(-1:0)
  end
  nc = length(its)

  println("Dimension is $d");
  minimas = zeros(d); maximas = zeros(d);
  covmatrix = zeros(d,npoints);
  v_tot = [1];
  q_tot = [1];
  C = sparse([1],[1],[1]);
  grid = 0;
  s = 1
  J = ones(n,1);
  #-#-#-#-#-#-#-#-#-#-#--#-#-##-#-#-#-#
  for i in 1:d
    println("Dimension $i/$d")
    minimas[i] = minimum(X[:,i]); maximas[i] = maximum(X[:,i]);
    grid = collect(linspace(minimas[i],maximas[i],npoints-nc));
    dx = grid[2]-grid[1];
    for j in 1:nc
      unshift!(grid,grid[1]-dx);
    end
    for j in 1:npoints
      covmatrix[i,j] = kernels.compute([grid[1]],[grid[j]]);
    end
    toep = toeplitz(covmatrix[i,:]);
    (v,q) = eig(toep);
    closest_id = floor(Int64,(X[:,i]-grid[1])/dx)+1 #Closest value on dimension i
    closest_dist = (X[:,i]-grid[1])/dx-closest_id+1;
    closest_id = closest_id - its[nc];
    C_i = zeros(nx,nc) #Coeff indices
    if cubic
      for j in 1:nc
        C_i[:,j] = cubics_coeff(closest_dist+its[nc+1-j])
      end
    else
      for j in 1:nc
        C_i[:,j] = max(1-abs(closest_dist+its[nc+1-j]),0)
      end
    end
    J_i = zeros(nx,nc) #row indices
    for j in 1:nx
      J_i[j,:] = closest_id[j]+its
    end
    v_tot = kron(v,v_tot); q_tot = kron(q,q_tot);
    if i==1
      C = C_i; J = ones(n,1);
    else
      C = repeat(C,outer=[1,1,nc]) .* repeat(reshape(C_i,n,1,nc),outer=[1,size(C,2),1]);
      C = reshape(C,n,length(C)÷n);
    end
    J = repeat(J[:],outer=[1,nc]) + s*repeat(J_i-1,outer=[size(C,2)÷nc,1]);
    s = s*npoints;
  end
  I = repeat((1:n)',outer=[1,size(C,2)]);
  A = sparse(I[:],floor(Int64,J[:]),C[:],n,npoints^dim);
  v_tot += γ;
  K_UU = q_tot*diagm(v_tot)*(q_tot')
  invK_UU = q_tot*diagm(1./v_tot)*(q_tot')
  K_XU = A*K_UU
  #K_XX = A*K_UU*A'
  return (K_UU,invK_UU, K_XU)
end
basic_time = @elapsed  trueK = CreateKernelMatrix(X,kern.compute) + γ*eye(n)
basic_time += @elapsed invtrueK = Matrix(inv(Symmetric(trueK,:U)))
# basic_time += @elapsed trueK

new_time = @elapsed (K_UU,invK_UU, K_XU) = getKISSMatrices(X,kernels,γ;npoints=40)
  datatrue = heatmap(;z=trueK)
  dataapp = heatmap(;z=appK)
  datadiff = heatmap(;z=abs(trueK-K_XX))
  invdatadiff = heatmap(;z=abs(invappK-invtrueK))
  lay = Layout(;title="TrueK heatmap",yaxis=attr(autorange="reversed"))
  # display(plot([datatrue;dataapp],lay))
  #display(plot([datadiff],lay))
  # display(plot([invdatadiff],lay))
  println("Mean error is $(mean(abs(K_XX-trueK)))")
  println("Normal time : $basic_time \nNew time : $new_time")
# display(plot([gridpoints,points]))

# println("Computed in $firsttry s for toep and $basicinversion s for normal inversion ")
