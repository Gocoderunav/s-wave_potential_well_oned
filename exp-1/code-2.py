
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd

# s-wave schrodinger eqn solver
# for an infinite potential well

# defining functions  
def tridiagonal(n):
    # making the tridiagonal matrix
    #------------------------------------------------------------------------------------    
    rows = n
    colms = n
    mat_1 = np.zeros((rows,colms))      # n*n zero matrix
    v = pot_values(n) 
    delta_list = []
    for i in range(n):
        delta_list.append(m_star*(h_not**2)*v[i])
   
    # this is for making the tridiagonal matrix
    for i in range(rows): 
        for j in range(colms): 
            if((i == n-n) ):
                if(j==0): 
                    mat_1[i][j] = -2-delta_list[i]
                elif(j==1):
                    mat_1[i][j] = 1
            elif(i==n-1):
                if( (i == n-1)):
                    if(j==i):
                        mat_1[i][j] = -2-delta_list[i]
                    elif(j==i-1):
                        mat_1[i][j] = 1
            else: 
                mat_1[i][i-1] = 1
                mat_1[i][i] = -2-delta_list[i]
                mat_1[i][i+1] = 1
    return mat_1

#------------------------------------------------------------------
# this function returns a list of diagonal values of potential matrix

def pot_values(n):
    # V is dimensionless
    V = np.zeros((n))
    for i in range(n):
        V[i] = i*0  # for infinite well potential 
        
    return V 
        
#-------------------------------------------------------------------

def potential_mat(n):
    # now for potential matrix
    # V = 0    # 0 because it is an infinite potential well
    V = pot_values(n)
    rows = n
    colms = n
    mat_2 = np.zeros((rows,colms))       # n*n zero matrix
    for i in range(rows):
        for j in range(colms):
           
                mat_2[i][i] = V[i]
    return mat_2 
#-------------------------------------------------------------------------
# psi vector theoretically 
def psi(x,i):
    factor = 7  # for removing the overlap
    res = ((np.sqrt(2/1)*np.sin(i*(np.pi)*x/l))/factor)
    # this is to set the wavefunctions in energy levels
    res = res + ((i-1)*(i-1))
    return res
    
def psi_2(x,i):
    factor = 7  # for removing the overlap
    res = ((np.sqrt(2/1)*np.sin(i*(np.pi)*x/l))/factor)
    return res
    

#---------------------------------------------------------------------------




#############        MAIN PROGRAM           ###############################

# UNIVERSAL constants to be used
me = 9.1e-31                    # mass of electron in kg
h_cross = 6.62e-34/(2*(np.pi))   # reduced plank's constant in Js
 
#   -----inputs from the user----------------------------------------
# n = number of nodal points ((except the 2 end points,))
n = int(input('enter the number of nodal points: (50 will be better)'))
# end points of well are 'a','b' and the length of the well is
# 'l' in metres
l = float(input('enter the length of well in angestroms: '))
# m = mass of the system in consideration in kg
m = float(input('enter the mass of system in units of mass of electron: '))
#--------------------------------------------------------------------------

# to make quantities dimesnionless ------------------------------------------
l = l*(10**(-10))
alpha = l  
a = 0
b = a + l
# h_not = step size  (dimensionless) 
h_not = l/(n+1)        ## divide by alpha
h_not = h_not/alpha
#  m*= me/m = 1
m = m*me 
m_star = m/me
#-------------------------------------------------------------------------

 
# calling functions------------------and solving eigen value problem------------------------------------
tri_mat = tridiagonal(n)
pot_mat = potential_mat(n)

print('tridiagonal\n',tri_mat,'\n\n')         
print('potential mat\n',pot_mat,'\n\n')


mat_new = tri_mat - pot_mat   # subtracting the potential matrix from tridaigonal matrix

# using inbuilt func to get eigenvectors and eigenvalues
lamda,v=np.linalg.eig(mat_new)
# now we actually have to multiply by -1 to the eigenvalues 
# acc to our orignal problem
for i in range(n):
    lamda[i] = -lamda[i]
# now because python does not gives eigenvalues and eigenvectors 
# in an order hence we have to do it now 
 
#---------------------------------------------------------------------
# combo list here contains eigenvalues and corrosponding eigenvectors
combo_list = [] 
v = v.T  # transposing it to get psi on rows
for i in range(n):
    q = [lamda[i],v[i]]
    combo_list.append(q) 

# now ordering both lamda and v
lamda_new = lamda
lamda_new.sort() 
combo_new = [] 
v_new = []
for i in range(n): 
    for j in range(n):  
        if(combo_list[j][0]==lamda_new[i]):
            combo_new.append(combo_list[j])
            v_new.append(combo_list[j][1])
v_new = np.array(v_new)
# checking normalisation of psi
norm = np.matmul(v_new,v_new.T)
print('wavefunction squared\n',np.around(norm,2))
print('all wavefunctions are normalised\n')
    
#---------------------------------------------------------------------

# pinting the eigenvalues and eigenvectors respectively
m=5
if(n<10):
    m=n
for i in range(m):
    print('e-value old: ',np.around(combo_list[i][0],2))
    print('e-vector: ',np.around(combo_list[i][1],3))
    print('E-value new:', np.around(lamda_new[i],2)) 
    print('E-vector new', np.around(v_new[i],3))
    print('\n')
#-----------------------------------------------------------------------------
# now calculating energy 
#-----------------------------------------------------------------------------
# E_star = lamda/m_star*h_not sq
E_star = []
for i in range(n):
    E_star.append(lamda[i]/(m_star*h_not*h_not))
# E = E_not*E_star 
E = []
for i in range(n):
    # E.append((h_cross**2/(2*me*alpha*alpha))*E_star[i])    # for E in joules keep this only
    aa=((h_cross**2/(2*me*alpha*alpha))*E_star[i])           # for E in eV keep these 2 lines
    E.append(aa/(1.6*((10)**(-19))))


# theoretical energy 
E_th = []
for i in range(n):
    # E_th.append((((i+1)**2)*(np.pi)**2)*(h_cross**2/(2*m*alpha*alpha)))  # for E in joules keep this only
    aa=((((i+1)**2)*(np.pi)**2)*(h_cross**2/(2*me*alpha*alpha)))           # for E in eV keep these 2 lines
    E_th.append(aa/(1.6*((10)**(-19))))
    
# diving energy of every state by energy of ground state
E_by_E_ground = []
quantum_number = []
for i in range(n):
    x = E[i]/E[0]
    E_by_E_ground.append(E[i]/E[0]) 
    quantum_number.append(np.sqrt(x))
    
# printing eigenvalues,eigenvectors, energy in tabular form
print('\n\n')
data={'eigen value':np.around(lamda,2),'eigen energy (eV)':(E),'E theoretical value(eV)':E_th}
df=pd.DataFrame(data)    
print(df)
#------------------------------------------------------------------------------------

# now including the end points where psi is 0.
# here in the eigenvector matrix we are inserting end point values of psi
v_new = np.insert(v_new, 0, [0], axis=1)
v_new = np.insert(v_new, n+1, [0], axis=1)
#---------------------------------------------------------------------


###################   plots #############################
x_values = np.linspace(a, b ,n+2)

#-------------------------------------------------------------
# energy plot vs state of system
plt.scatter(np.arange(n), E,marker='^')
plt.xlabel('state')  
plt.ylabel('energy value (E) in eV')
plt.grid()
plt.show()
#---------------------------------------------------------------
# # plotting sq of the wave func as probability vs space
# x_values = np.linspace(a,b,n)
v_new = np.array(v_new) 
plt.plot(x_values,v_new[0]**2,label ='ground')
plt.plot(x_values,v_new[1]**2,label='first excited')
plt.plot(x_values,v_new[2]**2,label='second excited')
 
plt.xlabel(' space(in metre)',fontname='Franklin Gothic Medium',fontsize=15)
plt.ylabel('probability ( $\psi^2$)',fontname='Franklin Gothic Medium',fontsize=15)
plt.legend()
plt.grid()
plt.show()
#----------------------------------------------------------------

plt.title('numerical energy & theoretical energy vs state',fontname='Franklin Gothic Medium',fontsize=15)
plt.plot(E,'*',label='numerical energy(eV)',color='red')
plt.plot(E_th,'.',label='theoretical energy(eV)',color='green')
plt.grid() 
plt.legend()
plt.show()
#-----------------------------------------------------------------
# plotting nodal point vs quantum number
plt.title('nodal point(state) vs quantum number')
plt.plot(range(1,n+1),quantum_number,'*')
plt.xlabel('nodal point')
plt.ylabel('quantum number')
plt.grid()
plt.show()
#-----------------------------------------------------------------



# just for plotting we are modifying the co-ordinates for psi vector wrt energy levels
# v_new is the matrix containing psi vectors in the rows

psi_vec_num = []
for i in range(n):
    row = []
    for j in range(n+2):
        row.append(v_new[i][j] + ((i*i)))
    psi_vec_num.append(row)
    
x = np.linspace(a,b,15)

x_ticks = np.linspace(a-2*l,b+2*l,6)
y_ticks = [(i*i) for i in range(0,4)]
colors = ['orange','blue','green','red']
labels = ['ground $\psi$','first $\psi$','second $\psi$','third $\psi$']
n_p = 4
for i in range(n_p):
    plt.plot(x_values,psi_vec_num[i],label =labels[i],color=colors[i])
    if(i==0):
        plt.scatter(x,psi(x,i+1),marker='o',s=3,color='black',label='theoretical $\psi$')
    else:
        plt.scatter(x,psi(x,i+1),marker='o',s=3,color='black')  

plt.xlabel('1D space in metres',fontname='Franklin Gothic Medium',fontsize=15)
plt.title('wavefunction plotted wrt energy levels',fontname='Franklin Gothic Medium',fontsize=15)
plt.xticks(ticks = x_ticks)
plt.yticks(ticks = y_ticks) 
plt.legend()
plt.grid()
plt.show() 
 

#-------extra----------------------------------------------
fig,((ax1, ax2, ax3)) = plt.subplots(3, figsize=(17, 10))
fig.suptitle(' wavefunction $\psi$ ', fontsize=40)
 
for i in range(n_p):
    plt.subplot(4, 1, n_p-i)
    plt.grid()
    plt.plot(x_values,v_new[i],label =labels[i],color=colors[i])
    plt.plot(x,psi_2(x,i+1),'*',label='theoretical $\psi$')   
    plt.legend(loc='best') 
fig.show()






