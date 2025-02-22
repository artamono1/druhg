import warnings

try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn(e.args[0])
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    HAVE_CYTHON = False


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)

_druhg_unionfind = Extension('druhg._druhg_unionfind',
                         sources=['druhg/_druhg_unionfind.pyx'])
_druhg_tree = Extension('druhg._druhg_tree',
                         sources=['druhg/_druhg_tree.pyx'])
_druhg_group = Extension('druhg._druhg_group',
                         sources=['druhg/_druhg_group.pyx'])
_druhg_label = Extension('druhg._druhg_label',
                         sources=['druhg/_druhg_label.pyx'])

def readme():
    with open('README.rst', encoding='utf8') as readme_file:
        return readme_file.read()

def requirements():
    # The dependencies are the same as the contents of requirements.txt
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip()]

configuration = {
    'name': 'druhg',
    'version': '1.7.3',
    'description': 'Universal clustering based on dialectical materialism',
    'long_description': readme(),
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.8',
    ],
    'keywords': 'cluster clustering density dialectics',
    'url': 'https://github.com/artamono1/druhg',
    'maintainer': 'Pavel Artamonov',
    'maintainer_email': 'druhg.p@gmail.com',
    'license': 'BSD',
    'packages': ['druhg', 'druhg.tests'],
    'install_requires': requirements(),
    'ext_modules': [
                    _druhg_unionfind,
                    _druhg_tree,
                    _druhg_group,
                    _druhg_label
                    ],
    'zip_safe': False,
    'cmdclass': {'build_ext': CustomBuildExtCommand},
    'tests_require': ['pytest'],
    'data_files': ('druhg/_druhg_unionfind.pxd', 'druhg/_druhg_group.pxd', 'druhg/_druhg_tree.pxd',)
}

if not HAVE_CYTHON:
    warnings.warn('Due to incompatibilities with Python 3.7 druhg now'
                  'requires Cython to be installed in order to build it')
    raise ImportError('Cython not found! Please install cython and try again')

setup(**configuration)
