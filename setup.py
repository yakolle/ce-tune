from setuptools import setup

setup(name='ce-tune',
      version='0.1',
      description="Tune the ml model's superparameters",
      url='http://github.com/yakolle/ce-tune',
      author='Yakolle Zhang',
      license='MIT',
      scripts=['util','data_util','tune_util','cv_util'],
      zip_safe=False)
      
