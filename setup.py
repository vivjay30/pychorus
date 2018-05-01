from distutils.core import setup
setup(
  name = 'pychorus',
  packages = ['pychorus'], # this must be the same as the name above
  version = '0.1',
  description = 'Library to find the choruses and interesting segments of songs',
  author = 'Vivek Jayaram',
  author_email = 'vivekjayaram@gmail.com',
  url = 'https://github.com/vivjay30/pychorus', # use the URL to the github repo
  download_url = 'https://github.com/vivjay30/pychorus/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['Chorus', 'Audio Signal Processing', 'Music'], # arbitrary keywords
  install_requires=[
    'librosa',
    'numpy',
    'scipy',
    'soundfile',
    'matplotlib'
  ],
  classifiers = [],
)
