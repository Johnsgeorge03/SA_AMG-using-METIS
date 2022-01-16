import numpy as np
import scipy as sp
from scipy import sparse
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import metis
from matplotlib import collections
import pyamg

########################## TEST PROBLEMS ######################################

class TestPoisson:
    'HW 5 problem'
    data = sio.loadmat('square.mat')
    A = data['A'].tocsr()
    B = data['B']
    B = B.reshape((B.shape[0],))
    V = data['vertices'][:A.shape[0]] 
    Elmts = data['elements']
    
    x = V[:,0]
    y = V[:,1]
    
    
    
class TestRotatedAnisotropy:
    sten = pyamg.gallery.diffusion.diffusion_stencil_2d(type = 'FD', 
                                    epsilon = 0.01,theta = np.pi/4)
    A  = pyamg.gallery.stencil_grid(sten, (16,16), format = 'csr')
    B  = np.ones(A.shape[0])
    B2 = np.zeros((A.shape[0],2))
    B2[:,0]   = 1
    B2[::2,1] = 1
    X, Y = np.meshgrid(np.linspace(0,1,16),
                   np.linspace(0,1,16))
    x = X.reshape(-1,)
    y = Y.reshape(-1,)
    V = np.vstack((x,y)).T

################## END OF TEST PROBLEMS #######################################



############################ FUNCTION DEFINITIONS #############################

# def plotall(resid, pre_runs , post_runs):
#     '''
#     Creates a 3x3 subplot.
#     Resid = list of list of list
#     '''
#     k = -1
#     fig , ax = plt.subplots(3,3, figsize = (18,18))
#     fig.suptitle("Pre-smooth = %g, Post-smooth = %g"
#                  %(pre_runs, post_runs), fontsize = 20)
#     for I in ax:
#       for J in I:
#         k+=1
#         for i in range(len(resid[k])):
#           J.semilogy(resid[k][i], lw = 3,label='$\omega = $%g'%(omega_list[i]))
#         J.set_title(r"$\theta$ = %g"%(theta_list[k]))
#         J.set(xlabel='Iterations', ylabel='Residuals')
#         J.legend()
#         J.grid()
#     plt.show()
  
  

def plotmatrix(thismatrix, ax, lw=1, color='tab:blue'):
    # edges of the matrix graph #CS556 - DEMO
    E = np.vstack((thismatrix.tocoo().row,thismatrix.tocoo().col)).T  

    lines = np.empty((E.shape[0], 2, 2))
    lines[:,0,0] = x[E[:,0]] # xstart
    lines[:,1,0] = x[E[:,1]] # xend
    lines[:,0,1] = y[E[:,0]] # ystart
    lines[:,1,1] = y[E[:,1]] # yend

    ls = collections.LineCollection(lines, color=color)
    ax.add_collection(ls, autolim=True)
    ls.set_linewidth(lw)
    ax.set_title("metis recursive")
    ax.autoscale_view()
    ax.axis('off')




def strength(A, theta):
    '''
    Gives the strength matrix.
    |Aij| >= theta * (Aii * Ajj) ^ 0.5
    '''
    A = A.toarray()
    n = A.shape[0]
    S = np.zeros((n,n))
    for i in range(n):
      for j in range(n):
        if abs(A[i,j]) >= theta*np.sqrt(A[i,i]*A[j,j]):
          S[i,j] = 1
    return sparse.csr_matrix(S)




def partition(S):
    '''
    Aggretion using METIS.
    
    '''
    n = S.shape[0]
    adj_list = [] # adjacency list of the strength matrix
    for i in range(n):
        adj_list.append(tuple(S.getrow(i).indices))
    
    
    
    #npart1 = int(S.nnz/(1*spla.norm(S, ord = np.inf)))
    #assert(npart1 != 0)
    # no of partitions is manually given for 1 level using maxnagg
    # maxagg = 48  grid(16x16) rotated anisotropy
    # maxagg = 176 grid(32x32) rotated anisotropy
    # For poisson use maxnagg = int(np.sqrt(S.nnz))
    maxnagg = 48#int(np.sqrt(S.nnz))
    assert(maxnagg != 0)
    (edgecuts, parts) = metis.part_graph(adj_list, nparts = maxnagg, 
                     recursive = True)
  
    nagg = max(parts) +1
    print("nagg_metis: ", nagg, "S.nnz: ", S.nnz)
    AggOp = np.zeros((n,int(nagg)))
    for i in range(n):
        j = int(parts[i])  
        AggOp[i, j] = 1
    return sparse.csr_matrix(AggOp)



def aggregate(S):
    '''
    Conventional aggregation strategy - aggregate to the smallest.

    '''
    n       = S.shape[0]
    agg     = -1*np.ones(n)
    agg_num = 0
    i       = 0
    
    while(i<n):
      J = S.getrow(i).indices
      if(np.all(agg[J] == -1)):
        agg[J] = agg_num
        agg_num+=1
      if i == n - 1:
        unagg_node = np.where(agg==-1)[0]
        if unagg_node.size:
          for j in unagg_node:
            neighbour = S.getrow(j).indices
            smallest = np.inf
            for k in neighbour:
                if S.getrow(k).nnz <= smallest and agg[k] !=-1:
                  smallest = k
            agg[j] = np.copy(agg[smallest])
      i+=1
    
    nagg = max(agg) +1
    print("nagg: ",nagg, "S.nnz ", S.nnz)
    AggOp = np.zeros((n,int(nagg)))
    for i in range(n):
      j = int(agg[i])  
      AggOp[i, j] = 1
    
    return sparse.csr_matrix(AggOp)





def tentative(AggOp, B):
     
    if len(B.shape) == 1:
        Agg  = sparse.csr_matrix.copy(AggOp)
        Bcan = np.copy(B)
        nagg = Agg.shape[1]
        m    = Bcan.shape[0]
        T    = np.zeros((m,nagg))
        Bc   = np.zeros((nagg))
        for i in range(nagg):
          ag       = Agg.toarray()[:,i]
          T[: , i] = (ag*Bcan)
          vec_mag  = np.linalg.norm(T[:,i], ord = 2)
          T[:, i]  = T[:,i] / vec_mag
          Bc[i]    = vec_mag
        return sparse.csr_matrix(T), Bc

    else:
        Agg  = sparse.csr_matrix.copy(AggOp)
        Bcan = np.copy(B)
        nagg = Agg.shape[1]
        m    = Bcan.shape[0]
        n    = Bcan.shape[1]
        T    = np.zeros((m,n*nagg))
        Bc   = np.zeros((n*nagg,n))
        for i in range(nagg):
            ag = Agg.toarray()[:,i].reshape((-1,1))
            Q, R = np.linalg.qr((ag*Bcan))
            #print(Q,R)
            T[:, i*n:i*n+n] =  Q
            Bc[i*n:i*n+n,:] =  R
        return sparse.csr_matrix(T), Bc
            


def interp(T, A):
    'T is sparse'
    
    I     = np.eye(T.shape[0])
    D     = A.diagonal()**(-1)
    Dinv  = sparse.diags(D, format='csr')
    P     = (I-(2/3) * Dinv *A ) * T
    return sparse.csr_matrix(P)




def smoothed_aggregation(A,B,max_level, max_coarse, theta):
    AA = []
    PP = []
    ml = []
    l  = 0
    Al  = sparse.csr_matrix.copy(A)
    AA.append(Al)
    Bl  = np.copy(B)
    while(Al.size >= max_coarse and l <= max_level):
      S = strength(Al,theta)
      Agg = aggregate(S)
      T, Bl = tentative(Agg, Bl)
      P = interp(T, Al)
      Al = P.T*Al*P
      AA.append(Al)
      PP.append(P)
      l+=1
    
    ml.append(AA)
    ml.append(PP)
    return ml





def smoothed_aggregation_metis(A,B,max_level, max_coarse, theta):
    """
    Uses recursive partitioning of METIS.
    
    Parameters
    ----------
    A : MATRIX 
    B : CANDIDATE VECTOR
    theta : STRENGTH OF CONNECTION THRESHOLD
    cfg   : coarsening factor
    Returns
    -------
    ml : LIST WITH ML[0] HAVING LEVELS + 1 ,A MATRICES
        ML[1] HAVING LEVELS INTERPOLATION MATRICES
    
    """
    AA  = []
    PP  = []
    SS  = []
    AGG = []
    ml  = []
    l   = 0
    Al  = sparse.csr_matrix.copy(A)
    AA.append(Al)
    Bl  = np.copy(B)
    while(Al.size >= max_coarse and l <= max_level):
      S = strength(Al,theta)
      Agg = partition(S)
      T, Bl = tentative(Agg, Bl)
      P = interp(T, Al)
      Al = P.T*Al*P
      SS.append(S)
      AA.append(Al)
      PP.append(P)
      AGG.append(Agg)
      l+=1
    
    ml.append(AA)
    ml.append(PP)
    ml.append(SS)
    ml.append(AGG)
    return ml




def jacobi(A, x, b, nu):
    '''
    Weighted Jacobi for smoothing 
    nu - no.of runs
    '''
    D    = A.diagonal()**(-1)
    Dinv = sparse.diags(D, format='csr')
    for i in range(nu):
      x = x + omega * Dinv * (b - A * x)
    return x



def gauss_siedel(A, u0, f, nu):
    '''Gauss - Siedel relaxation'''
    u     = u0.copy()
    DE    = sparse.tril(A, k = 0, format = 'csc')
    F     = -sparse.triu(A, k = 1, format = 'csc')
    DEinv = spla.inv(DE)
    for run in range(nu):
      u  = DEinv@F@u + DEinv@f
    return u




def vcycle(ml, b, x0, levels, pre_run, post_run):
    
    x       = np.copy(x0)
    bl      = np.copy(b)
    levels  = levels
    xlevels = []
    blevels = []
    ## DOWN CYCLE ##
    for i in range(levels):
      x  = jacobi(ml[0][i], x, bl, pre_run) #pre-smoothing
      xlevels.append(np.copy(x))
      blevels.append(np.copy(bl))
      bl = ml[1][i].T*(blevels[i] - ml[0][i] * xlevels[i])
      x  = np.zeros(bl.shape)
    
    ## COARSE SOLVE ##
    Ac = ml[0][levels]
    blevels.append(bl)
    xlevels.append(sparse.linalg.spsolve(Ac,bl))
    
    ## UP CYCLE ##  
    for i in range(levels, 0, -1): #up
      x   = xlevels[i]
      bl  = blevels[i]
      xf  = xlevels[i-1]
      xlevels[i-1] = xf + ml[1][i-1] * x
      xf  = jacobi(ml[0][i-1], xf, blevels[i-1], post_run)  #post-smoothing
    
    return xlevels[0]




def solve(ml, x0, b, tol, maxiter,levels, 
          pre_run, post_run, omega, residuals = None):
      
    x = np.copy(x0)
    assert levels <= len(ml[1])
    for i in range(maxiter):
      x = vcycle(ml, b, x, levels = levels, 
                 pre_run = pre_run , post_run =post_run)
      residuals.append(sp.linalg.norm(b - A * x))
      if residuals[i] <= tol:
        break
    return x, residuals

################## END OF FUNCTION DEFINITIONS ###############################



######################## SOLVE PROBLEM ########################################
A       = TestRotatedAnisotropy.A
B       = TestRotatedAnisotropy.B
omega   = 2/3
theta   = 0.15
pre     = 2
post    = 0
level   = 1
u0      = np.random.rand(A.shape[1])
ml      = smoothed_aggregation_metis(A,B,max_level=0, 
                                      max_coarse=10, theta=theta)

# ml      = smoothed_aggregation(A,B,max_level=0, 
#                                       max_coarse=10, theta=theta)
b       = np.zeros(A.shape[0])
u, res  = solve(ml, u0,b, tol=1e-6, maxiter = 200, levels = level,
                pre_run = pre, post_run = post, omega = omega, residuals=[])


############### PRINTS CONVERGENCE FACTOR & PLOTS RESIDUALS ##################
#print("res:")
#print(res)
res = np.array(res)
print("CF:\n")
print(res[1:]/res[:-1])
print(np.linalg.norm(b-A@u))
plt.figure(figsize=(9,9))
plt.semilogy(res, lw = 3)
plt.title(r"$\omega = %g, \theta = %g$, metis, level = %g, \
          grid = 16x16"%(omega, theta, level), fontsize = 20)#change title here
plt.xlabel("Iterations", fontsize = 20)
plt.ylabel("Residuals", fontsize = 20)
plt.grid()

###################### PLOT AGGREGATES #######################################
x = TestRotatedAnisotropy.x
y = TestRotatedAnisotropy.y
V = TestRotatedAnisotropy.V
f, ax = plt.subplots(1, figsize=(6,6))
S = strength(A,theta = 0.15)
plotmatrix(A, ax)
plotmatrix(S, ax, lw=2)
AggOp = partition(S) #aggregates from METIS partitioning
#AggOp = aggregate(S)  #aggregates from aggregate to smallest


# CS 556 DEMO
for i in range(AggOp.shape[1]):
    J = AggOp.getcol(i).tocoo().row
    for j1 in J:
        for j2 in J:    
            if j1 != j2:
                if A[j1, j2]:
                    plt.plot([x[j1], x[j2]], [y[j1], y[j2]], 'r', lw=4)
    if len(J) == 1:
        plt.plot(x[J[0]], y[J[0]], 'ro', ms=10)
                   
# for i, v in enumerate(V):
#     plt.text(v[0], v[1], '%d'%i, fontsize=10)


# change title here  
plt.title("Metis recursive bisection. No.of aggregates %g"%(AggOp.shape[1]))