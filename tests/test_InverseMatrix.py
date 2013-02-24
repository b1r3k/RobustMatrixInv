'''
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/.

Created on Oct 7, 2011

@author: b1r3k

'''

import unittest

import numpy
from scipy.sparse import rand
from scipy.linalg import inv

import matrix_inv

class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass



    def test_SingleRowInversionConcept(self):
        """
        Test koncepcji odwracania macierzy wierszami
        Na tej podstawie jest zbudowana implementacja z wykorzystaniem wywolan MAP
        """
        
        A = rand(100, 100, format = "csc", density = 1.0)
        
        sp_res = matrix_inv.single_row_sparse_inv(A)
        std_inv_res = inv(A.todense())
        
        self.assertTrue( numpy.allclose(std_inv_res, sp_res.todense()) )
            
            
        pass
    
    def testSimpleMatrix(self):
        """
        One-by-one inversion vs. Mapped inversion
        Threading: ON
        For small chunks
        """
        
        ndim = 100
        
        A = rand(ndim, ndim, format = "csc", density = 1.0)
        
        std_inv_res = inv(A.todense())
        sparselu_inv_res = matrix_inv.sparselu_inv(A, True, 10)
        
        self.assertTrue( numpy.allclose(sparselu_inv_res.todense(), std_inv_res) )

        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()