from setuptools import setup, find_packages

setup(
    name="transfer",
    version="0.5.0",
    description="Transfer learning for deep image classification",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['numpy', 'keras', 'tensorflow', 'pyyaml', 'tqdm', 'pandas', 'opencv-python', 'termcolor', 'colorama'],
    python_requires='>=3',

    # metadata for upload to PyPI
    author="Matthew Sochor",
    author_email="matthew.sochor@gmail.com",
    license="MIT",
    keywords="keras transfer learning resnet deep neural net image classification command line",
    url="http://github.com/matthew-sochor/transfer",   # project home page, if any
    download_url="http://github.com/matthew-sochor/transfer",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
    entry_points = {
        'console_scripts': [
            'transfer = transfer.__main__:main'
        ]
    }
)
