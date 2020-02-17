import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

##################################
# Generate some random pictures
##################################
NPIX_X = 20

def dft_harmonic(k, dim):
    x = np.array([[(i, j) for j in range(dim[1])] for i in range(dim[0])])
    return np.real(np.exp(2*np.pi*1j*np.tensordot(x, np.divide(k, dim), 1)))

#discr = np.linspace(0, 1, NPIX_X)
#pic1 = np.array([[np.sin(10*((x-0.5)**2+(y-0.5)**2))**2 for y in discr] for x in discr])
#pic2 = np.array([[np.sin(10*((x-0.5)**2-(y-0.5)**2))**2 for y in discr] for x in discr])
pic1 = dft_harmonic((0,1), (NPIX_X, NPIX_X))
pic2 = dft_harmonic((2,5), (NPIX_X, NPIX_X))
pic1 /= np.linalg.norm(pic1)
pic2 /= np.linalg.norm(pic2)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(pic1)
ax[1].imshow(pic2)
plt.show()

%%time
##################################
# Write down update rules
##################################

def pack_vars(x, U, W):
    return np.hstack((x, U.flatten(), W.flatten()))

def unpack_vars(package, shp):
    x = package[:shp[1]]
    U = package[shp[1]:shp[1]+np.prod(shp)].reshape(shp)
    W = package[shp[1]+np.prod(shp):].reshape(np.flip(shp))
    return x, U, W
    
def rhs_x(x, I, U, W, p):
    return W.dot(I - U.dot(x)) / p['tau_x']

def rhs_U(x, I, U, p):
    return (np.outer(I, x) - U.dot(np.outer(x, x))) / p['tau_U'] 

def rhs_W(x, I, W, p):
    return (np.outer(x, I) - np.outer(x, x).dot(W)) / p['tau_U'] 
    
def rhs_xUW(package, I, p):
    x, U, W = unpack_vars(package, p['U_SHAPE'])
    xdot = rhs_x(x, I, U, W, p)
    Udot = rhs_U(x, I, U, p)
    Wdot = rhs_W(x, I, W, p)
    return pack_vars(xdot, Udot, Wdot)

##################################
# Initialize model
##################################

param = {'dt' : 0.0001, 'tau_x' : 0.001, 'tau_U' : 0.1, 'U_SHAPE' : (NPIX_X**2, 2)}

# Input
I_lst = [pic1.flatten(), pic2.flatten()] 
get_input = lambda t, tau: I_lst[int(t / tau) % 2]

#############
# Training
#############

T_TRAINING = 5 # Seconds
TAU_INPUT_TRAINING = 0.01
N_STEP_TRAINING = int(T_TRAINING / param['dt'] / 10)
t_training = np.linspace(0.0, T_TRAINING, N_STEP_TRAINING+1)

# Initial Values
x0 = np.random.uniform(0.4, 0.6, param['U_SHAPE'][1])
U0 = np.random.uniform(0, 1, np.prod(param['U_SHAPE'])).reshape(param['U_SHAPE'])
W0 = np.random.uniform(0, 1, np.prod(param['U_SHAPE'])).reshape(np.flip(param['U_SHAPE']))
U0 /= np.linalg.norm(U0)
W0 /= np.linalg.norm(U0)
y0 = pack_vars(x0, U0, W0)

# Forwards integration
#print("Training...")
print("Training for", N_STEP_TRAINING, "steps...")
rhs_training = lambda y,t : rhs_xUW(y, get_input(t, TAU_INPUT_TRAINING), param)
rez_training = integrate.odeint(rhs_training, y0, t_training)
# rez_training = integrate.odeint(rhs_training, y0, [0, T_TRAINING], hmin=param['dt'])

# Compute transposition error
err_transposition = []
for t, y in zip(t_training, rez_training):
    _, U, W = unpack_vars(y, param['U_SHAPE'])
    err_transposition.append(np.linalg.norm(U - W.transpose()))

#############
# Testing
#############

T_TESTING = 0.4 # Seconds
TAU_INPUT_TESTING = 0.1
N_STEP_TESTING = int(T_TESTING / param['dt'])
t_testing = np.linspace(0.0, T_TESTING, N_STEP_TESTING+1)

# Forwards integration
print("Testing for", N_STEP_TESTING, "steps...")
_, U_Testing, W_Testing = unpack_vars(rez_training[-1], param['U_SHAPE'])
rhs_testing = lambda x,t : rhs_x(x, get_input(t, TAU_INPUT_TESTING), U_Testing, W_Testing, param)
rez_testing = integrate.odeint(rhs_testing, x0, t_testing)

# Compute prediction error
err_prediction = []
for t, x in zip(t_testing, rez_testing):
    I = get_input(t, 0.1)
    err_prediction.append(np.linalg.norm(I - U_Testing.dot(x)))

#############
# Plotting
#############

# Plot receptive fields
fig, ax = plt.subplots(nrows=2, ncols=2, tight_layout=True)
ax[0][0].imshow(U_Testing[:,0].reshape((NPIX_X, NPIX_X)))
ax[0][1].imshow(U_Testing[:,1].reshape((NPIX_X, NPIX_X)))
ax[1][0].imshow(W_Testing[0,:].reshape((NPIX_X, NPIX_X)))
ax[1][1].imshow(W_Testing[1,:].reshape((NPIX_X, NPIX_X)))
ax[0][0].set_title("U1")
ax[0][1].set_title("U2")
ax[1][0].set_title("W1")
ax[1][1].set_title("W2")
plt.show()

# Plot Transpose errors
plt.figure()
plt.plot(t_training, err_transposition)
plt.show()

# Plot representation errors and values
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].plot(t_testing, rez_testing[:, 0], label='tex1')
ax[0].plot(t_testing, rez_testing[:, 1], label='tex2')
ax[0].legend()
ax[0].set_title("representation value")
ax[1].plot(t_testing, err)
ax[1].set_title("representation error")
plt.show()