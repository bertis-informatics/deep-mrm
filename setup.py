import os
from setuptools import find_packages, setup

requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
requirements = []
dependency_links = []
with open(requirements_file) as fh:
    for line in fh:
        line = line.strip()
        if line:
            # Make sure the github URLs work here as well
            split = line.split('@')
            split = split[0]
            split = split.split('/')
            url = '/'.join(split[:-1])
            requirement = split[-1]
            requirements.append(requirement)
            # Add the rest of the URL to the dependency links to allow
            # setup.py test to work
            if 'git+https' in url:
                dependency_links.append(line.replace('git+', ''))

setup(
    name='DeepMRM',
    version='0.2.0',
    description='Automated quantification for targeted proteomics',
    author='Jungkap Park / Bertis',
    author_email='jungkap.park@bertis.com',
    packages=find_packages(
        exclude=[
            'deepmrm.train', 
            'deepmrm.evaluation', 
            'deepmrm.data_prep']),
    # packages=find_namespace_packages(),
    # package_dir={"": "src"},
    include_package_data=True,
    data_files=[
        # Each (directory, files) pair in the sequence 
        # specifies the installation directory and the files to install there.
        ('', ['config.yml']),
        # ('models', ['models/DeepMRM_Model.pth']),
    ],
    keywords=[
        'proteomics', 
        'mass spectrometry',
        'MRM',
        'biomarker', 
        'machine learning', 
        'object detection',
    ],
    python_requires='>=3.8',
    license='',
    url='https://gitlab.com/bertis-informatics/deep-mrm',
    # install_requires=requirements,
    # dependency_links=dependency_links,
)
