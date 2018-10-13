import os
from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

install_requires = []
dependency_links = []
for entry in required:
    if "git+" in entry:
        dependency_links.append(entry)
    else:
        install_requires.append(entry)

for entry in dependency_links:
    os.system("pip install {}".format(entry))

setup(name='humblerl',
      version='0.0.1',
      install_requires=install_requires)
