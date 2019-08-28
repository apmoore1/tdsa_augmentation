from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='tdsa_augmentation',
      version='0.0.1',
      description='TDSA Augmentation',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/apmoore1/tdsa_augmentation',
      author='Andrew Moore',
      author_email='andrew.p.moore94@gmail.com',
      license='Apache License 2.0',
      install_requires=[
          'allennlp>=0.8.3',
          'gensim'
      ],
      python_requires='>=3.6.1',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3.6'
      ])