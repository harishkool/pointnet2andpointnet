from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='ballquery',
      ext_modules=[CUDAExtension('torch_ballquery', ['torch_ballquery.cpp','torch_ballquery_cuda.cu'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        include_dirs=["."])],
       cmdclass={'build_ext':BuildExtension})