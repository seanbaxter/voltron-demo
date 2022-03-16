#include <cstdio>
#include <cuda_runtime_api.h>

// The general definition.
constexpr int f(int x) {
  return 1;
}

// The definition used during constant evaluation.
[[override::consteval]] 
int f(int x) {
  return 2;
}

// The definition used by CUDA codegen.
[[override::nvvm]]
int f(int x) {
  return 3;
}

// A more specific definition used by CUDA codegen.
[[override::nvvm(__nvvm_arch >= (nvvm_arch_t)75)]]
int f(int x) {
  return 4;
}

// An AMDGPU definition.
[[override::amdgpu]]
int f(int x) {
  return 5;
}

// Constant evaluation chooses the [[override::consteval]] definition.
constexpr int x = f(1);
@meta printf("constexpr: f(1) = %d\n", x);

__global__ void kernel(int i) {
  // A kernel chooses the most specific matching [[override::nvvm]] definition.
  @meta for enum(nvvm_arch_t arch : nvvm_arch_t) {
    if target(arch == __nvvm_arch) {
      // We're targeting a specific arch.
      printf("%s: f(1) = %d\n", arch.string, f(i));
    }
  }
}

int main(int argc, char** argv) { 
  // A call from the host chooses the general definition.
  printf("primary: f(1) = %d\n", f(argc));

  // Launch the kernel. That will choose the most specific override for 
  // the NVVM arch.
  kernel<<<1, 1>>>(1);
  cudaDeviceSynchronize();
}