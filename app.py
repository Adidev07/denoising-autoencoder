"""Train an encoder-decoder neural network to reconstruct clean signals from noisy observations."""
from kit import *
TITLE='Denoising Autoencoder'
LIVE=False

def dataset(seed,n=500):
    rng=np.random.default_rng(seed);t=np.linspace(0,2*np.pi,16);weights=rng.normal(0,.5,(n,3));basis=np.array([np.sin(t),np.cos(t),np.sin(2*t)]);return weights@basis
def init():
    rng=np.random.default_rng(7);return [rng.normal(0,.15,(16,5)),np.zeros(5),rng.normal(0,.15,(5,16)),np.zeros(16)]
def forward(p,x):h=np.tanh(x@p[0]+p[1]);return h,h@p[2]+p[3]
def loss_grad(p,x,target):
    h,pred=forward(p,x);err=pred-target;dz=2*err/err.size;dh=(dz@p[2].T)*(1-h*h)
    return float(np.mean(err**2)),[x.T@dh,dh.sum(axis=0),h.T@dz,dz.sum(axis=0)]
def train(clean,epochs=500):
    rng=np.random.default_rng(9);p=init();history=[]
    for _ in range(epochs):
        noisy=clean+rng.normal(0,.35,clean.shape);loss,g=loss_grad(p,noisy,clean);history.append(loss)
        for v,grad in zip(p,g):v-=.3*grad
    return p,history
def run(d,a):
    clean=dataset(a.seed);test=dataset(a.seed+1,200);noisy=test+np.random.default_rng(5).normal(0,.35,test.shape);p,h=train(clean);pred=forward(p,noisy)[1]
    return dict(noisy_mse=float(np.mean((noisy-test)**2)),denoised_mse=float(np.mean((pred-test)**2)),initial_training_loss=h[0],final_training_loss=h[-1],bottleneck=5),dict(**{f'parameter_{i}':v for i,v in enumerate(p)},clean=test[:20],noisy=noisy[:20],reconstruction=pred[:20])

if __name__=="__main__":main(run)
