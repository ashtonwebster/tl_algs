# Transfer Learning Algorithm Library

[Wikipedia](https://en.wikipedia.org/wiki/Inductive_transfer) defines transfer learning as "a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem." Basically, this just means training using two or more domains and attempting to extend the predictions of these *source* domains to a new, previously unseen, *target* domain. This library serves as an implementation of several transfer learning algorithms.

## Implemented Algorithms

* [Burak](https://doi.org/10.1007/s10664-008-9103-7): Use the k closest source instances to each target instance (a.k.a. NN filtering) [1]
* [Peters](https://doi.org/10.1109/MSR.2013.6624057): Find closest target instances to training instances [2]
* [TrBagg](https://doi.org/10.1109/ICDM.2009.9): Bootstrap sample to create many learners, then filter [3]
* [Transfer Naive Bayes (TNB)](https://doi.org/10.1016/j.infsof.2011.09.007): Modification of Naive Bayes using gravitational weighting [4]
* [Transfer Component Analysis (TCA)](https://doi.org/10.1109/TNN.2010.2091281): Use a projection method to bring source and target domains to a common space [5]
* [Improved Transfer Component Analysis (TCA+)](https://doi.org/10.1109/ICSE.2013.6606584): Similar to TCA, but uses normalization before applying the projection [6]
* Baselines:
    * Hybrid (all available source AND target training instances)
    * All available target training
    * All available source domain training

## Installing

For best results, install [Anaconda](https://www.continuum.io/downloads).  This will (a) ensure you have all of the necessary dependencies and (b) allow you to view the example as a jupyter notebook.

```
# install the TCA dependency
> git clone https://github.com/ashtonwebster/transferlearning
> cd transferlearning/code/python
# install the dependency
> pip install -e .
> cd ../../../
# clone the repository
> git clone https://github.com/ashtonwebster/tl_algs.git
> cd tl_algs
# install the package!
> python setup.py install
```

Another option is to install so that (local) updates to the source will immediately affect the installed package:

```
pip install -e .
```

Now you can use tl_algs in any file by including:
```
import tl_algs
```
At the top of the file.

## Example

See examples/example1.ipynb or examples/example1.html for an example usage.

## References

[1] Turhan, B., Menzies, T., Bener, A. B., & Di Stefano, J. (2009). On the relative value of cross-company and within-company data for defect prediction. Empirical Software Engineering, 14(5), 540–578. https://doi.org/10.1007/s10664-008-9103-7

[2] Peters, F., Menzies, T., & Marcus, A. (2013). Better cross company defect prediction. In IEEE International Working Conference on Mining Software Repositories (pp. 409–418). https://doi.org/10.1109/MSR.2013.6624057

[3] Kamishima, T., Hamasaki, M., & Akaho, S. (2009). TrBagg: A simple transfer learning method and its application to personalization in collaborative tagging. Proceedings - IEEE International Conference on Data Mining, ICDM, 219–228. https://doi.org/10.1109/ICDM.2009.9

[4] Ma, Y., Luo, G., Zeng, X., & Chen, A. (2012). Transfer learning for cross-company software defect prediction. Information and Software Technology, 54(3), 248–256. https://doi.org/10.1016/j.infsof.2011.09.007

[5] Pan, S. J., Tsang, I. W., Kwok, J. T., & Yang, Q. (2011). Domain Adaptation via Transfer Component Analysis. IEEE Transactions on Neural Networks, 22(2), 199–210. https://doi.org/10.1109/TNN.2010.2091281

[6] Nam, J., Pan, S. J., & Kim, S. (2013). Transfer defect learning. In Proceedings - International Conference on Software Engineering (pp. 382–391). https://doi.org/10.1109/ICSE.2013.6606584
