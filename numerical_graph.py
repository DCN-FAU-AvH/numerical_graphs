#!/usr/bin/python3

# Copyright 2023 Dragos Manea. All rights reserved

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.animation import FuncAnimation

#Define the tree via oriented edges

V=6 #number of vertices

V_in=[1,2] #input vertices

V_out=[5,6] #output edges

V_int=[3,4] #interior edges

E=[(1,3,2,0.6),(2,3,2,0.4),(3,4,3,1),(4,5,2,0.3),(4,6,3,0.7)] #(initial node, final node, length, speed) # al edges are parametrised by [0,length]


d=2 #diffusivity

#Numerical parameters
h=0.05
tau=0.005 #satisfying tau<h/(max speed)

TIME=20 #the in-game time

#The initial data on every edge

ctt=0.10539922456186433 # a constant used to define the initial data

u0=[lambda x: x/2*ctt, lambda x: x-2+ctt, lambda x: np.exp(-(x-1.5)**2), lambda x: ctt+x, lambda x: np.sin(x*np.pi)+ctt*(x+1)] #the initial data -- continuous at junctions, for each edge parametrized as [0,length]

#The Dirichlet boundary data on each input and output vertex

ub_in=[lambda t: t, lambda t: (t+1)*(ctt-2)] #the time-dependent boundary conditions, for each input vertex
ub_out=[lambda t: ctt+2+t,lambda t: 4*ctt] #the time-dependent boundary conditions, for each output vertex


# _____ _   _ ____     ____ ___  _   _ _____ ___ ____ 
#| ____| \ | |  _ \   / ___/ _ \| \ | |  ___|_ _/ ___|
#|  _| |  \| | | | | | |  | | | |  \| | |_   | | |  _ 
#| |___| |\  | |_| | | |__| |_| | |\  |  _|  | | |_| |
#|_____|_| \_|____/   \____\___/|_| \_|_|   |___\____|

#The in/out edges of the internal vertices

E_in=[[] for v in V_int] #the vector of incoming edges, corresponding to each interior vertex
E_out=[[] for v in V_int] #the vector of outgoing edges, corresponding to each interior vertex

for i in range(0,len(V_int)):
    for j in range(0,len(E)):
        if E[j][0]==V_int[i]: #if the start of an edge is the current vertex, add it to the corresonding intry in E_out
            E_out[i].append(j) 
        if E[j][1]==V_int[i]: #if the end of an edge is the current vertex, add it to the corresonding intry in E_in
            E_in[i].append(j)

#The discretised initial data

Ndim=0 #the dimension of the discrete vector
offset=np.zeros(len(E)+1,dtype=np.int32) #the offset of each edge in the final vector
for i in range(0,len(E)):
    offset[i]=Ndim
    Ndim+=int(E[i][2]/h) #discretise every edge
offset[-1]=Ndim #the offset of the vertex data
Ndim+= len(V_int)

U0=np.zeros(Ndim) #will contain the discretised initial data

for i in range(0,len(E)): #populate the edge data
    for j in range(0,offset[i+1]-offset[i]):
        U0[offset[i]+j]=(u0[i])(j*h) #the numerical location of the j-th point on the i-th edge

for i in range(0,len(V_int)):
    U0[offset[-1]+i]=u0[E_out[i][0]](0) #the numerical location of the i-th vetex


#Plot the function on a particular graph

fig = plt.figure(figsize=(30,15),dpi=80) #controls the size of the plot
ax = fig.add_subplot(1, 1, 1, projection='3d') #we want a 3D plot for showing the function on the graph 

def plot_particular(U):
    ax.clear() #This function can be animated
    #Display the example graph in the plane
    P3=(0,0) #One of the points in  the graph
    W=np.array((np.cos(np.pi/6),np.sin(np.pi/6))) #direction of an edge
    WBar=np.array((np.cos(np.pi/6),-np.sin(np.pi/6)))

    #Computation of the position of the other vertices
    P1=P3+E[0][2]*(-WBar)
    P2=P3+E[1][2]*(-W)
    P4=P3+E[2][2]*np.array([1,0])
    P5=P4+E[3][2]*W
    P6=P4+E[4][2]*WBar
    P=[P1,P2,P3,P4,P5,P6]
    P_int=[P3,P4] #interior vertices
    Px=[p[0] for p in P]
    Py=[p[1] for p in P]

    #Plot the edges
    E_plot=[]
    for e in E:
        Ee=[[P[e[0]-1][0],P[e[1]-1][0]],[P[e[0]-1][1],P[e[1]-1][1]]]
        E_plot.append(Ee)
        ax.plot(Ee[0],Ee[1],color="black")
    #Plot the vertices
    ax.scatter(Px,Py,color="blue")

    #Label the vertices
    for i in range(0,len(P)):
        ax.text(Px[i],Py[i],-0.25,'V'+str(i+1))

    ax.set_zlim(-0.5,3)

    def conv_comb(X,Y,tc): #the comvex combination of two points X,Y with parameter tc
        return X*(1-tc)+Y*tc

    for i in range(0,len(E)): #populate the edge data
        toPlot=[]
        for j in range(0,offset[i+1]-offset[i]):
            tc=j/(offset[i+1]-offset[i]) #the relative position on the segment of the current point
            Ee=E_plot[i]
            toPlot.append([conv_comb(Ee[0][0],Ee[0][1],tc),conv_comb(Ee[1][0],Ee[1][1],tc),U[offset[i]+j]]) #interpolate between the two vertices and get the current point. Plot the value in the values vector
        ax.plot3D([tp[0] for tp in toPlot],[tp[1] for tp in toPlot],[tp[2] for tp in toPlot]) #plot one edge at  a time
    
    for i in range(0,len(V_int)):
        ax.scatter3D(P_int[i][0],P_int[i][1],U[offset[-1]+i]) #plot the values in the inner vertices

steps=int(TIME//tau) #the number of time steps

#some useful parameters
r=tau*d/h**2 
s=tau/h

#The matrix for the implicit scheme

def get_index(A,e): #get the index of element e in array A, or -1 if e is not present
    try:
        return A.index(e)
    except ValueError:
        return -1

T=sp.lil_matrix((Ndim,Ndim),  dtype=np.float64) #use sparse for memory optimisation

for i in range(0,len(E)): #construct the part of matrix corresponding to the edges
    for j in range(0,offset[i+1]-offset[i]):
        T[offset[i]+j,offset[i]+j]=1+2*r
        if (j!=0):
            T[offset[i]+j,offset[i]+j-1]=-r-s/2*E[i][3] 
        else: #if we are at the beginning of an edge, use the the value in the vertex for computation
            ind=get_index(V_int,E[i][0])
            if (ind!=-1):
                T[offset[i]+j,offset[-1]+ind]=-r-s/2*E[i][3]
        if (j!=offset[i+1]-offset[i]-1):
            T[offset[i]+j,offset[i]+j+1]=-r+s/2*E[i][3]
        else: #if we are at the end of an edge, use the the value in the vertex for computation
            ind=get_index(V_int,E[i][1])
            if (ind!=-1):
                T[offset[i]+j,offset[-1]+ind]=-r+s/2*E[i][3]

for i in range(0,len(V_in)): #construct the part of the matrix concerning vertices
    T[offset[-1]+i,offset[-1]+i]=1+2*r
    for e in E_in[i]: #the inpact of the incoming edges
        T[offset[-1]+i,offset[e+1]-1]=(-2*r-s*E[e][3])/(len(E_in[i])+len(E_out[i]))
    for e in E_out[i]: #the inpact of the outgoing edges
        T[offset[-1]+i,offset[e]]=(-2*r+s*E[e][3])/(len(E_in[i])+len(E_out[i]))

T=T.tocsc() #convert T in a better format for solving systems

#The vector of boundary data

def gen_boundary_data(t): #Creates the time-dependent vector used to encode the Dirichlet boundary conditions
    B=np.zeros(Ndim)
    for j in range(0,len(V_in)):
        for i in range(0,len(E)):
            if (E[i][0]==V_in[j]):
                B[offset[i]]=-(ub_in[j](t))*(r+s/2*E[i][3]) #for input vertices, use the corresponding ub_in

    for j in range(0,len(V_out)):
        for i in range(0,len(E)):
            if (E[i][1]==V_out[j]):
                B[offset[i+1]-1]=(ub_out[j](t))*(-r+s/2*E[i][3]) #for input vertices, use the corresponding ub_out
    return B

U=U0
Us=[U0] #will contain the solution at each time step
for i in range(0,steps): #iterate through timpe
    B=gen_boundary_data(i*tau)
    U=sp.linalg.spsolve(T,U-B) #solve the implicit system
    Us.append(U) #append the current solution to the vector

def animate(i): #animate the plot
    plot_particular(Us[i])
    ax.set_title('Time t='+str(i*tau),y=1,pad=-2)

ani = FuncAnimation(fig, animate, frames=steps, interval=50, repeat=False) #the parameter interval controls the speed of the animation

plt.show()