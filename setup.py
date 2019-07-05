from setuptools import setup, find_packages

setup(
    name="BookRecommendation",
    version="1.0",
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'sklearn', 'nltk'],
    package_data={'BookRecommendation': ['inputs/*.*']},
    zip_safe=False)
