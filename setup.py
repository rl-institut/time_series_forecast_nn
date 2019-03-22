from distutils.core import setup

setup(
    name='TimeSeriesForecastNerualNet',
    version='1.0',
    packages=['timeseriesforecastneuralnet',],
    license='MIT',
    long_description=open('README.md').read(),
    author='Reiner Lemoine Institut',
    url='https://github.com/rl-institut/time_series_forecast_nn',
    install_requires=[
        'numpy',
        'tensorflow',
        'matplotlib',
      ],
    python_requires='<=3.6',
)