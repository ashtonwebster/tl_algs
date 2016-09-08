from setuptools import setup

setup(name='tl_algs',
      version='0.1',
      description='A library of transfer learning algorithms for python',
      url='https://para.cs.umd.edu/purtilo/transfer-learning-algorithm-library/tree/master',
      author='Ashton Webster',
      author_email='ashton.webster@gmail.com',
      license='MIT',
      packages=['tl_algs'],
      install_requires=[
          'pandas',
          'sklearn',
          'matplotlib',
          'numpy'
        ],
      zip_safe=False)
