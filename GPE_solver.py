#!/usr/bin/python
# -*- coding: utf-8 -*-
########
__author__ = "Neng-Chun Allen Chiu"
__copyright__ = "Copyright 2020, GPE solver"
__credits__ = ["Neng-Chun Allen Chiu", "C.A. Chen"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "C.A. Chen"
__email__ = "acechen@cirx.org"
__status__ = "Prototype"
########
import numpy as np
import numexpr as ne
import scipy as sp
import mkl_fft as fft


def GPE_2d(p, V, g, psi0, imag_t=True):
    x = np.linspace(-p['X_range'], p['X_range'], p['X_grid'])
    X, Y = np.meshgrid(x, x)
    dx = x[1]-x[0]                        # dx is the X mesh spacing
    k = sp.fft.fftfreq(p['X_grid'], dx)*2.0*np.pi # k values, used in the fourier spectrum analysis
    Kx, Ky = np.meshgrid(k, k)
    
    N = int(p['T']/p['dt'])               # number of steps
    Ntstore = int(p['T']/p['snap'])+2     # number of stored results
    Ntskip = int(p['snap']/p['dt'])       # skip without saving
    Tstore = np.arange(0, p['T'], p['snap'])        # time of results
    psi_out = np.zeros((Ntstore, p['X_grid'], p['X_grid']), complex)
    psi_out[0,:] = psi0
    energy_out = np.zeros(Ntstore, complex)
    ep = np.zeros(Ntstore)                # store convergence test result
    
    dt = p['dt']
    error = p['error']
    
    ## For real time propagation
    if imag_t is False:
        prefactor = 1j
        
    ## For imaginary time propagation
    else:
        prefactor = 1

    U1 = -prefactor*V*dt/2.0
    C1 = -prefactor*g*dt/2.0
    K = (Kx**2+Ky**2)
    Kin = np.exp(-prefactor*K*dt/2.0)
    Kinjj = np.exp(-1j*K*dt/2.0)
    psi = psi0

    for i in range(N):
        # Split step Fourier transform
        C = C1[i]  # lookup 
        U = U1[i]  # lookup
        
        #psi = np.exp(U+C*psi*np.conj(psi))*psi
        psi = ne.evaluate('exp(U+C*psi*conj(psi))*psi')
        psi = fft.ifft2(Kin*fft.fft2(psi))
        #psi = np.exp(U+C*psi*np.conj(psi))*psi
        psi = ne.evaluate('exp(U+C*psi*conj(psi))*psi')
        
        if imag_t:
            # For Imaginary time propagation, normalize every loop
            psi_int = np.sum(np.conj(psi)*psi)*dx*dx
            psi = psi/psi_int**0.5

        if i%Ntskip == 0:
            # Store results
            j = int(i/Ntskip)
            
            if imag_t:
                # Calculate the energy
                
                #psi_new = np.exp(1j*U+1j*C*psi*np.conj(psi))*psi
                psi_new = ne.evaluate('exp(1j*U+1j*C*psi*conj(psi))*psi')
                psi_new = fft.ifft2(Kinjj*fft.fft2(psi_new))
                #psi_new = np.exp(1j*U+1j*C*psi_new*np.conj(psi_new))*psi_new
                psi_new = ne.evaluate('exp(1j*U+1j*C*psi_new*conj(psi_new))*psi_new')
                energy_out[j+1] = np.log(np.sum(np.conj(psi_new)*psi)*dx*dx)*1j/dt
            
                # Calculate how much the solution has converged.
                ep[j] = np.abs(energy_out[j+1]-energy_out[j])
                psi_out[j+1,:] = psi

                if ep[j] <= error:
                    # If the absolute error is less than specified, stop now.
                    psi_out = psi_out[0:j+2]
                    break
            else:
                # Store the wavefuction in psi_out
                psi_out[j+1,:] = psi
    
    return psi_out, energy_out, ep, Tstore

def V_ho(p):
    # harmonic trap
    x = np.linspace(-p['X_range'], p['X_range'], p['X_grid'])
    X, Y = np.meshgrid(x, x)
    return 0.5*(p['trap_f']/w_s(p))**2*(X**2+Y**2)

def V_box(p):
    # box trap
    x = np.linspace(-p['X_range'], p['X_range'], p['X_grid'])
    X, Y = np.meshgrid(x, x)
    return np.tanh(-p['b_size']-X)+np.tanh(X-p['b_size'])+np.tanh(-p['b_size']-Y)+np.tanh(Y-p['b_size'])
	
def V(p):
    # time dependent potential
    t = np.arange(0, p['T'], p['dt'])
    
    if p['trap'] == 'ho':
        V_0 = V_ho(p)
    if p['trap'] == 'box':
        V_0 = V_box(p)
    
    if p['mod_v']:
        V_t = np.sin(t)+1
    else:
        V_t = np.ones(t.shape)
    
    return np.tensordot(V_t, V_0, axes=0)

def g0(p):
    # coupling constant in dimensionless GP
    x = np.linspace(-p['X_range'], p['X_range'], p['X_grid'])
    X, Y = np.meshgrid(x, x)
    return 0*X + 0*Y + 4*np.pi*a_s(p)/x_s(p)
    
def g(p):
    # time dependent interaction
    t = np.arange(0, p['T'], p['dt'])
    g_0 = g0(p)
    
    if p['mod_g']:
        g_t = np.sin(t)
    else:
        g_t = np.ones(t.shape)
    
    return np.tensordot(g_t, g_0, axes=0)

def psi_init(p, imag_t=True):
    # initial wavefunction
    x = np.linspace(-p['X_range'], p['X_range'], p['X_grid'])
    X, Y = np.meshgrid(x, x)
    
    if imag_t:
        # initial guess
        psi = np.exp(-(X**2+Y**2)/2)
    else:
        # initial wavefunction with a kick
        psi = np.exp(-(X**2+Y**2)/2)/np.sqrt(np.pi)*np.exp(1j*2*X)
    
    return psi.astype(complex)
    
def Prob(psi):
    # probability
    return np.real(psi*np.conj(psi))

def t_s(p):
    # scaling parameter t' = t/t_s
    return 1/p['wx']
    
def x_s(p):
    # scaling parameter x' = x/x_s
    return np.sqrt(1.054e-34/2.206e-25/p['wx'])
    
def E_s(p):
    # scaling parameter E' = E/E_s
    return 1.054e-34*p['wx']
    
def w_s(p):
    # scaling parameter w' = w/w_s
    return p['wx']
    
def a_s(p):
    # s-wave scattering length in metre
    return 5.291e-11*p['a']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    parameters = {
        'X_range':        20, # grid range
        'X_grid':        256, # grid number
        'T':            1e-2, # total evolution time in s
        'dt':           1e-6, # evolution time step in s
        'snap':         1e-4, # snapshot in s
        'error':        1e-4, # tolerence for convergence
        'trap':        'box', # trap type
        'trap_f': 2*np.pi*20, # harmonic trap frequency in rad/s
        'b_size':          5, # box trap range
        'a':             100, # scattering length in Bohr radius
        'wx':    2*np.pi*2e3, # vertical trap frequency in rad/s
        'mod_v':       False, # modulate potential
        'mod_g':       False, # modulate interaction
    }
    
    V = V(parameters)
    g = g(parameters)
    psi0 = psi_init(parameters, imag_t=True)
    start = time.time()
    psi_gs, energy_gs, ep, _ = GPE_2d(parameters, V=V, g=g, psi0=psi0, imag_t=True)
    psi_f, _, _, Tstore = GPE_2d(parameters, V=V, g=g, psi0=psi_gs[-1], imag_t=False)
    end = time.time()
    print('runtime = %.3f' % (end-start))
    plt.figure()
    plt.imshow(Prob(psi_gs[-1]))
    plt.figure()
    plt.imshow(Prob(psi_f[-2]))
    plt.show()