# Transfer Learning Algorithm Library

[Wikipedia](https://en.wikipedia.org/wiki/Inductive_transfer) defines transfer learning as "a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem." Basically, this just means training using two or more domains and attempting to extend the predictions of these *source* domains to a new, previously unseen, *target* domain. This library serves as an implementation of several transfer learning algorithms.

## Implemented Algorithms

* [Burak](https://doi.org/10.1007/s10664-008-9103-7): Find close target instances to training instances
* [Peters](https://doi.org/10.1109/MSR.2013.6624057): Use the k closest source instances to each target instance
* [TrBagg](https://doi.org/10.1109/ICDM.2009.9): Bootstrap sample to create many learners, then filter
* [Gravity Weighted](https://doi.org/10.1016/j.infsof.2011.09.007)
* Baselines:
    * Hybrid (all available source AND target training instances)
    * All available target training
    * All available source domain training

## Installing

```
# clone the repository
> git clone https://para.cs.umd.edu/purtilo/transfer-learning-algorithm-library
> cd tl_algs
# install the package!
> python setup.py install
```

Now you can use tl_algs in any file by including:
```
import tl_algs
```
At the top of the file.

## Example

See examples/example1.ipynb or examples/exampl1.html for an example usage.
