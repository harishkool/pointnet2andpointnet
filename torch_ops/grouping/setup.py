from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='grouping',
      ext_modules=[CUDAExtension('torch_grouping', ['torch_grouping.cpp','torch_grouping_cuda.cu'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        include_dirs=["."])],
       cmdclass={'build_ext':BuildExtension})