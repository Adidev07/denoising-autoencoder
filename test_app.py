import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_gradient(self):
        x=dataset(2,4);p=init();_,g=loss_grad(p,x,x);v=p[2][0,0];eps=1e-5;p[2][0,0]=v+eps;hi=loss_grad(p,x,x)[0];p[2][0,0]=v-eps;lo=loss_grad(p,x,x)[0];self.assertAlmostEqual((hi-lo)/(2*eps),g[2][0,0],places=6)
    def test_training_improves(self):
        _,h=train(dataset(2,200),200);self.assertLess(h[-1],h[0])
    def test_shapes(self):self.assertEqual(forward(init(),dataset(2,10))[1].shape,(10,16))
