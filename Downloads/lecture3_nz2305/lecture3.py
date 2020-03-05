
# coding: utf-8

# In[ ]:
import numpy as np
display_mode = 'Plot'
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

def hp_memristor(t, V, memristor, MOD_A = 'Switch',MOD_B = 'Hard'):
    if not(MOD_B == 'Soft' or MOD_B == 'Hard'):
        print('input error')
        return
    t = np.reshape(t,(1,t.size))
    V = np.reshape(V,(1,V.size))
    
    # beta reciprocal
    br = 1 / memristor['beta']
    # Initialize the time-related parameters
    dt = np.diff(t[0,0:2])
    N = t.shape[1] - 1    
    
    # Initialize internal state, current and mem-conductance arrays
    x = np.zeros([1,N+1])
    I = np.zeros([1,N+1])
    M = np.zeros([1,N+1])
    x[0,0] = memristor['x0']
    
    # define M(x) function handle
    calc_m = lambda x: x + (1-x)*memristor['r']
    # define Hard Switch function handle
    f_hard = lambda i,x,dt: x + dt * (i * br) * x * (1-x)
    # define Soft Switch function handle
    f_soft = lambda i,x,dt: x + dt*(i * br)
    
    # Choose the specified switch method
    if MOD_B == 'Hard':
        f = f_hard
    else:
        f = f_soft    

    for i in range(N):
        M[0,i] = calc_m(x[0,i])
        I[0,i] = V[0,i] / M[0,i]
        x[0,i+1] = f(I[0,i], x[0,i],dt)
    
    
    M[0,N] = calc_m(x[0,N])
    I[0,N] = V[0,N] / M[0,N]
    
    M[0,N] = calc_m(x[0,N])
    I[0,N] = V[0,N] / M[0,N]        
    return I,x


# In[ ]:


def plot_memristor(t,I,V,x,display_mode = 'plot',FontSize = 12):    
    t = np.reshape(t,(1,t.size))
    I = np.reshape(I,(I.shape[0],I.shape[1]))
    V = np.reshape(V,(V.shape[0],V.shape[1]))
    x = np.reshape(x,(x.shape[0],x.shape[1]))
    
    #plot -b inline -f svg -r 96 -s 1280,900
    fig_width = 10
    fig_height = 17.78
    fig = plt.figure(facecolor = 'w',
                    figsize=(fig_width,fig_height),dpi = 72)
    
    default_fontsize = 12;
    #default_markersize = max([floor(mon_pos(4)/160),7]);
    
    isMultInput = (len(I) > 1)
    dt = np.diff(t[0:2])
    from scipy import integrate    
    Q = integrate.cumtrapz(I,t)
    F = integrate.cumtrapz(V,t)
    M  = V / I;
    
    if isMultInput:
        max_I = 1;
        max_V = 1;
        max_x = 1;
        max_M = 1;
    else:
        max_I = np.nanmax(I,axis = 1);
        max_V = np.nanmax(V,axis = 1);
        max_x = np.nanmax(x,axis = 1);
        max_M = np.nanmax(M,axis = 1);
    
    
    if isMultInput:
        #from matplotlib import cm
        ax1 = fig.add_axes([0.1, 0.85, 0.35, 0.1])
        ax1.set_title('Current and Voltage', fontsize = FontSize)
        ax1.set_ylabel('Input Voltage [V]', fontsize = FontSize)
        ax1.plot(t.T,V.T,linewidth = 2)
        
        ax2 = fig.add_axes([0.1, 0.7, 0.35, 0.1])
        ax2.set_xlabel('Time [s]', fontsize = FontSize)
        ax2.set_ylabel('Output Current [A]', fontsize = FontSize)
        ax2.plot(t.T,I.T,linewidth = 2)
        
        ax3 = fig.add_axes([0.6, 0.85, 0.35, 0.1])
        ax3.set_title('Internal State and Memristance', fontsize = FontSize)
        ax3.set_ylabel('State [A * s]', fontsize = FontSize)
        ax3.plot(t.T,x.T,linewidth = 2)
        
        ax4 = fig.add_axes([0.6, 0.7, 0.35, 0.1])
        ax4.set_xlabel('Time [s]', fontsize = FontSize)
        ax4.set_ylabel('Memristance [Ohm]', fontsize = FontSize)
        ax4.plot(t.T,M.T,linewidth = 2)
        
    else:
        ax1 = fig.add_axes([0.1,0.7,0.35,0.22])
        ax1.set_title('Normalized  Current and Voltage', fontsize = FontSize)
        ax1.set_ylabel('Current [A] and Voltage [V]', fontsize = FontSize)
        ax1.set_xlabel('Time [s]', fontsize = FontSize)
        ax1.set_xlim(t[0,0],t[0,-1])
        ax1.set_ylim(-1.1,1.1)        
        ax1.plot(t.T,I.T/max_I,'-r',linewidth = 2, label = 'I')
        ax1.plot(t.T,V.T/max_V,'-b',linewidth = 2, label = 'V')
        leg1 = ax1.legend(loc = 'upper right')
        
        ax2 = fig.add_axes([0.6,0.7,0.35,0.22])
        ax2.set_title('Normalized Internal State and Memristance', fontsize = FontSize)
        ax2.set_ylabel('State [A * s] and Memristance [Ohm]', fontsize = FontSize)
        ax2.set_xlabel('Time [s]', fontsize = FontSize)
        ax2.set_xlim(t[0,0],t[0,-1])
        ax2.set_ylim(0,1.1)        
        ax2.plot(t.T,x.T/max_x,'-r',linewidth = 2, label = 'Internal State')
        ax2.plot(t.T,M.T/max_M,'-b',linewidth = 2, label = 'Memristance')
        leg2 = ax2.legend(loc = 'upper right')
    
    
    ax5 = fig.add_axes([0.1, 0.4, 0.35, 0.22])
    ax5.set_ylabel('Current [A]', fontsize = FontSize)
    ax5.set_xlabel('Voltage [V]', fontsize = FontSize)
    ax5.set_title('Voltage and Current', fontsize = FontSize)
    ax5.plot(V.T,I.T,linewidth = 2)
    
    ax6 = fig.add_axes([0.6, 0.4, 0.35, 0.22])
    ax6.set_ylabel('Current [A]', fontsize = FontSize)
    ax6.set_xlabel('Memristance [Ohm]', fontsize = FontSize)
    ax6.set_title('Memristance and Current', fontsize = FontSize)
    ax6.plot(I.T,M.T,linewidth = 2)
        
    ax7 = fig.add_axes([0.1, 0.1, 0.35, 0.22])
    ax7.set_ylabel('Charge [A * s]', fontsize = FontSize)
    ax7.set_xlabel('Flux [W]', fontsize = FontSize)
    ax7.set_title('Charge and Flux', fontsize = FontSize)
    ax7.plot(100*F.T,100*Q.T,linewidth = 2)
    
    ax8 = fig.add_axes([0.6, 0.1, 0.35, 0.22])
    ax8.set_ylabel('Charge [A * s]', fontsize = FontSize)
    ax8.set_xlabel('Memristance [Ohm]', fontsize = FontSize)
    ax8.set_title('Memristance and Charge', fontsize = FontSize)
    ax8.plot(100*Q.T,M.T[:-1],linewidth = 2)


# In[ ]:


def SixPlots(t,I,V,state,M,state_name):    
    dt = t[0,1] - t[0,0]
    from scipy import integrate
    q = (dt*integrate.cumtrapz(I)).T;
    f = (dt*integrate.cumtrapz(V)).T;
    
    t = np.reshape(t,(t.size,1))            
    I = np.reshape(I,(I.size,1))            
    V = np.reshape(V,(V.size,1))    
    state = np.reshape(state,(state.size,1))    
    M = np.reshape(M,(M.size,1))            
        
    fig = plt.figure(facecolor = 'w',
                    figsize=(17.78,10),dpi = 72)
    
    plt.subplot(6,2,1)
    plt.plot(t,I,color = 'blue',linewidth = 2);    
    plt.xlabel('Time [s]',fontsize = 9)
    plt.ylabel('Current [{\mu}A]', fontsize = 9)
    plt.subplot(6,2,3)      
    plt.plot(t,V,color = 'blue',linewidth = 2);
    plt.xlabel('Time [s]',fontsize = 9)
    plt.ylabel('Voltage [mV]', fontsize = 9)
    
    plt.subplot(3,2,2)
    plt.plot(t,state,color='blue',linewidth = 2, label = ('Internal State ' + str(state_name)))
    plt.plot(t,M,color='maroon',linewidth = 2, label = 'Memconductance [mS]')
    plt.xlabel('Time [s]',fontsize = 9) 
    plt.legend(loc='upper right')
    
    plt.subplot(3,2,3)
    plt.plot(I,V,color='blue',linewidth = 2)
    plt.xlabel('Current [{\mu}A]',fontsize = 9)
    plt.ylabel('Voltage [mV]',fontsize = 9)
    
    plt.subplot(3,2,4)
    plt.plot(V,M,color = 'blue',linewidth = 2)   
    plt.xlabel('Voltage [mV]',fontsize = 9)
    plt.ylabel('Memconductance [mS]',fontsize = 9)
    
    plt.subplot(3,2,5)
    plt.plot(q,f,color = 'blue',linewidth = 2)
    plt.xlabel('Charge [fC]')
    plt.ylabel('Flux [nV*s]')
    
    
    plt.subplot(3,2,6)    
    plt.plot(f,M[:-1],color = 'blue',linewidth = 2)
    plt.xlabel('Flux [nV*s]')
    plt.ylabel('Memconductance [mS]')


# In[ ]:


def hodgkin_huxley(t,I_ext):
    t = 1000 * t
    dt = np.diff(t[0,0:2])
    
    E = np.array([-12,115,10.613])
    g = np.array([36,120,0.300])
    x = np.array([0,0,1.000])

    V = np.zeros((1,t.shape[1]))
    V[0,0] = -10
    I = np.zeros((t.shape[1],3))

    a = np.zeros((1,3))
    b = np.zeros((1,3))

    gnmh = np.zeros((t.shape[1],3))
    for i in range(1,t.shape[1]):
        a[0,0] = (10-V[0,i-1]) / (100 * (np.exp((10-V[0,i-1])/10) - 1))
        a[0,1] = (25-V[0,i-1]) / (10 * (np.exp((25-V[0,i-1])/10) - 1))
        a[0,2] = 0.07*np.exp(-V[0,i-1]/20)

        b[0,0] = 0.125 * np.exp(-V[0,i-1]/80)
        b[0,1] = 4 * np.exp(-V[0,i-1] /18)
        b[0,2] = 1 / (np.exp((30 - V[0,i-1])/10) + 1)

        tau = 1 / (a+b)
        x0 = a*tau

        x = (1-dt/tau) * x + dt/tau * x0

        gnmh[i,0] = g[0]*x[0,0]**4
        gnmh[i,1] = g[1]*x[0,1]**3 * x[0,2]
        gnmh[i,2] = g[2]

        I[i,:] = (gnmh[i,:] *(V[0,i-1]-E))
        V[0,i] = V[0,i-1] + dt * (I_ext[0,i] - sum(I[i,:]))
                 
    return V, I, gnmh


# In[ ]:


def hhn_w_states(t,I_ext):
    t = 1000 * t
    dt = np.diff(t[0,0:2])
    
    E = np.array([-12,115,10.613])
    g = np.array([36,120,0.300])
    x = np.array([0,0,1.000])

    V = np.zeros((1,t.shape[1]))
    V[0,0] = -10
    I = np.zeros((t.shape[1],3))

    a = np.zeros((1,3))
    b = np.zeros((1,3))

    gnmh = np.zeros((t.shape[1],3))
    xnmh = np.zeros((t.shape[1],3))
    
    for i in range(1,t.shape[1]):
        a[0,0] = (10-V[0,i-1]) / (100 * (np.exp((10-V[0,i-1])/10) - 1))
        a[0,1] = (25-V[0,i-1]) / (10 * (np.exp((25-V[0,i-1])/10) - 1))
        a[0,2] = 0.07*np.exp(-V[0,i-1]/20)

        b[0,0] = 0.125 * np.exp(-V[0,i-1]/80)
        b[0,1] = 4 * np.exp(-V[0,i-1] /18)
        b[0,2] = 1 / (np.exp((30 - V[0,i-1])/10) + 1)

        tau = 1 / (a+b)
        x0 = a*tau

        x = (1-dt/tau) * x + dt/tau * x0

        gnmh[i,0] = g[0]*x[0,0]**4
        gnmh[i,1] = g[1]*x[0,1]**3 * x[0,2]
        gnmh[i,2] = g[2]
        
        xnmh[i,0] = x[0,0]
        xnmh[i,1] = x[0,1]
        xnmh[i,2] = x[0,2]
                
        I[i,:] = (gnmh[i,:] *(V[0,i-1]-E))
        V[0,i] = V[0,i-1] + dt * (I_ext[0,i] - sum(I[i,:]))                 
    return V, I, gnmh, xnmh


# In[ ]:


def spike_detect(v):
    logi_1 = (v[0,1:-1]>v[0,0:-2])
    logi_2 = (v[0,1:-1]>v[0,2:])
    logi_3 = (v[0,1:-1]>10)
    
    logic_mat = [False]
    for i in range(len(logi_1)):        
        logic_mat.append((logi_1[i]) and (logi_2[i]) and (logi_3[i]))
    logic_mat.append(False)
    logic_mat = np.array(logic_mat)
    return logic_mat

