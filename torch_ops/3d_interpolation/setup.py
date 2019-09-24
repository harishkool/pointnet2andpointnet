from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='3dinterpolation',
      ext_modules=[CUDAExtension('torch_3dinterpolate', ['torch_3dinterpolate.cpp','torch_3dinterpolate_cuda.cu'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        include_dirs=["."])],
       cmdclass={'build_ext':BuildExtension})