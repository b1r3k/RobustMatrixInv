'''
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/.

Created on Oct 7, 2011

@author: b1r3k

'''
import sys
import logging

logger = logging.getLogger('default')

import numpy as np
from scipy.sparse.linalg import splu, spsolve
import scipy.sparse as sparse

# get it from http://www.connellybarnes.com/code/python/threadmap
from ThreadMap import map as threadmap

def single_row_sparse_inv(A):
    """
    This is proof of concept of method described here: http://home.ubalt.edu/ntsbarsh/Business-stat/otherapplets/SysEq.htm
    """
    col_size = A.shape[0]

    result = []
    
    rhs = np.zeros((col_size, 1))
    
    for i in xrange(0, col_size):
        rhs[i, 0] = 1.0
        X_result = sparse.csc_matrix(spsolve(A, np.ravel(rhs))).T
        
        result.append(X_result)
        rhs[i, 0] = 0.0
    
    inverted_mtx = sparse.hstack(result, format = "csc")
    
    assert inverted_mtx.shape == A.shape
        
    return inverted_mtx

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def sparselu_inv(A, use_threading = False, chunk_size = 5000):
    """
    This function uses concept shown in single_row_sparse_inv()
    The difference is that:
    1) it uses sparse linear algebra module i.e. splu
    2) tries to minimize peak memory usage
    3) divides matrix into chunks which are computed by threads
    
    @param A: square matrix
    @param use_threading: are we using ThreadMap module?
    @param column_chunk_size: how many columns should chunks have? can be used to optimize: execution time and/or peak memory usage
    """
    col_size = A.shape[0]
    row_size = A.shape[1]
    
    chunk_matrix_size = (chunk_size * row_size * sys.getsizeof(float)) / 1024 / 1024.
    
    logger.debug("sparselu_inv: use_threading = %s" % use_threading)
    logger.debug("sparselu_inv: chunk_size = %s => chunk_matrix size [ %d x %d ] = %d (Mb)" % (chunk_size, chunk_size, row_size, chunk_matrix_size))
    
    
    # A = L * U
    logger.debug("sparselu_inv: faktoryzacja splu(A)..")
    splu_factorized = splu(A)
    
    del A
    
    pool_map = []
    
    # splu_factorized.solve returns dense matrix as array
    # therefore, matrix is transformed into sparse one ASAP
    
    sparse_solver_wrapper = lambda rhs: sparse.csc_matrix(splu_factorized.solve(np.ravel(rhs.todense()))).T
    
    logger.debug("sparselu_inv: preparing RHS vectors..")
    for i in xrange(0, col_size):
        rhs = np.zeros((col_size, 1))
        rhs[i, 0] = 1.0
        
        pool_map.append(sparse.csc_matrix(rhs))
    
    logger.debug("sparselu_inv: Processing chunk pool ...")
    
    chunks_to_process = chunks(pool_map, chunk_size)
    amount_of_chunks = len(pool_map) / chunk_size
    
    for i, chunk in enumerate(chunks_to_process):
        if use_threading:
            results = threadmap(sparse_solver_wrapper, chunk, dynamic=False)
        else:
            results = map(sparse_solver_wrapper, chunk)

                        
        logger.debug("sparselu_inv: HStack join of chunk %d / %d.." % (i + 1, amount_of_chunks))
        
        chunk_results = sparse.hstack(results, format = "csc")
        
        if i:
            inverted = sparse.hstack((inverted, chunk_results), format = "csc")            
        else:
            inverted = chunk_results
            
    return inverted
            
if __name__ == '__main__':
    import unittest

    test_suite = unittest.defaultTestLoader.discover("tests", pattern='test*.py')

    unittest.TextTestRunner(verbosity=2).run(test_suite)
