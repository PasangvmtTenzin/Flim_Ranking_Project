from setuptools import setup, find_packages

setup(
    name='flim_ranking',
    version='0.1',
    description='A package for analyzing film ranking data.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/PasangvmtTenzin/Flim_ranking',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'seaborn',
        'matpotlib.pyplot'
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'flim_ranking=src.main:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)
