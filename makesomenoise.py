
def makesomenoise(xi, acf, fs):
    '''
    Produces noise with a specified autocorrelation function
    
    Parameters:
    
    xi   - input noise sequence
    acf  - desired autocorrelation of the noise as a numpy array
    fs.  - sample frequency of the autocorrelation and the noise
    
    Returns:
    xo   - reordered version of xi to produce the desired acf
    '''
    
    import numpy as np

    N = xi.shape[0]          # Get number of samples
    psd = np.fft.fft(acf, N) # Calculate PSD (Power Spectral Density) from ACF
    psd[0] = 0               # Zero out the DC component (remove mean)
    
    Af = np.sqrt(2 * np.pi * fs * N * psd) # Convert PSD to Fourier amplitudes
    mx = np.mean(xi)         # Calculate mean of samples 
    x  = xi - mx             # Make zero mean
    xs = np.sort(x)          # Store sorted signal xs with correct PDF
    k  = 1 
    idxr = np.zeros(N)       # Reference index array
    while(k != 0):
        Fx  = np.fft.fft(x)  # Compute FT of noise
        Px  = np.arctan2(np.imag(Fx), np.real(Fx))  # Get phases
        # Create a signal with correct PSD and original phases
        xg  = np.real(np.fft.ifft((np.exp(1.j*Px)) * np.abs(Af))) 
        idx = np.argsort(xg) # Get rank indices of signal with correct PSD
        x[idx] = xs          # Put original noise samples in desired rank order. 
        k = k+1              # Increment counter
        if (idx == idxr).all() :
            print(f'Number of iterations = {k}')
            k = 0            # If we converged, stop
        else:
            if k%500 == 0: print(f'Number of iterations = {k} ...')
            idxr = idx       # Re-set ordering for next iter
    x = x + mx               # Restore the original mean
    return(x)





def plot_noise(xi, xo, pdf, acf, fs, range1):
    '''
    Helper function to plot the output noise and targets.
    '''
    
    import matplotlib.pyplot as plt 
    import numpy as np
    
    if 'fig' in locals():
        plt.close(fig)
    fig = plt.figure(figsize = (8, 8))
    
    dt  = 1/fs
    N   = xi.size
    M   = acf.size - 1
    
    ax1 = plt.subplot(2,1,1)
    if xi.size>1000:
        time = np.arange(1000)*dt
        ax1.plot(time, xi[:1000], 'C0', label='input')
        ax1.plot(time, xo[:1000], 'C1', label='output')
    else:
        time = np.arange(xi.size)*dt
        ax1.plot(time, xi, 'C0', label='input')
        ax1.plot(time, xo, 'C1', label='output')        
    ax1.legend()
    ax1.set_xlabel('time (s)')
    ax1.set_title('noise sequency')

    ax2 = plt.subplot(2,2,3)
    ax2.hist(xo, bins=range1, density=True, color='C0', label='output')
    ax2.plot(range1, pdf, color='C1', linestyle='--', label='target')
    ax2.hist(xi, bins=range1, density=True, label='input', color='C3', alpha=0.2)
    ax2.legend()
    ax2.set_xlabel('amplitude (AU)')
    ax2.set_title('pdf')

    ax3 = plt.subplot(2,2,4)
    range2 = int(M/2)
    lags = np.linspace(-range2-1,range2,2*range2+1)
    xcor = np.correlate(xo, xo, mode='full')/N
    xcor = xcor/xcor.max()
    ax3.plot(lags*dt, xcor[N-range2-1:N+range2], color='C0', label='output')
    ax3.plot(lags*dt, acf, color='C1', linestyle='--', label='target') 
    ax3.legend()
    ax3.set_xlabel('lag (s)')
    ax3.set_title('acf')