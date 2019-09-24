from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='sampling',
      ext_modules=[CUDAExtension('torch_sampling', ['torch_sampling.cpp','torch_sampling_cuda.cu'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        include_dirs=["."])],
       cmdclass={'build_ext':BuildExtension})