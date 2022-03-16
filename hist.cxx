#include <cuda_runtime.h>
#include <map>
#include <cstdio>

__global__ void hist(int count, int range) {
  // Generate count number of random numbers on the device.
  // Feed them through std::set. This means just one of each number
  // is kept, and duplicates are removed.
  std::map<int, int> counters;
  for(int i : count)
    counters[rand() % range]++;

  // Print the unique, sorted elements to the terminal.
  printf("%d unique values generated:\n", counters.size());
  for(const auto& entry : counters)
    printf("%2d: %3d\n", entry.first, entry.second);
}

int main() {
  int count = 50;
  int range = 25;

  hist<<<1, 1>>>(count, range);
  cudaDeviceSynchronize();
}