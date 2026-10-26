from setuptools import setup

setup(name='ldsc',
      version='1.0.1.post3',
      description='LD Score Regression (LDSC) - optimized fork, Python 3 port',
      url='http://github.com/bulik/ldsc',
      author='Brendan Bulik-Sullivan and Hilary Finucane',
      author_email='',
      license='GPLv3',
      packages=['ldscore'],
      scripts=['ldsc.py', 'munge_sumstats.py'],
      python_requires='>=3.9',
      install_requires=[
            'numpy>=1.26,<1.27',
            'pandas>=1.5,<1.6',
            'scipy>=1.10,<1.11',
            'bitarray>=2.6,<2.7',
      ],
      extras_require={
            'test': ['nose>=1.3,<1.4'],
            'annot': ['pybedtools>=0.9'],  # make_annot.py only
      }
)
