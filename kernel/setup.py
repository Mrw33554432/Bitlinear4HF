from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='optimized_bitlinear',
    ext_modules=[
        CUDAExtension('optimized_bitlinear', [
            'matmul_cuda.cpp',
            'matmul_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
