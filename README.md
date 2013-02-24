RobustMatrixInv
===============

Alternative implementation of inverse matrix operation.

Problem
===============

Using functions provided by numpy like linalg.inv, splu and others did not help to solve on matrices limited memory resources.
Implementations which I was using (almost everything that was available in SciPy at that time) were having problems with memory allocation for big matrices. Usually given function was crashing at the initialization stage or during computations.
As solving time was less important for me than having solution, I decided to try implement inverse operation by myself.

Solution
===============

Basic idea was to split matrix and solve each chunk seperately [1]. The idea is simple and whatsmore, it can be fairly easy parallelized [2] so that execution time won't be much more devastating than by using native SciPy implementations.

Reference
===============
[1] http://home.ubalt.edu/ntsbarsh/Business-stat/otherapplets/SysEq.htm
[2] http://www.connellybarnes.com/code/python/threadmap

License
===============
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US"><img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">RobustMatricInv</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="http://about.me/lukasz.jachym" property="cc:attributionName" rel="cc:attributionURL">Lukasz Jachym</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/deed.en_US">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/b1r3k/RobustMatrixInv" rel="dct:source">https://github.com/b1r3k/RobustMatrixInv</a>.