# Treat CUDA code like regular C++ code
circle roman.cxx -sm_75
circle hist.cxx -sm_75
circle qsort.cxx -sm_75
circle virtual.cxx -sm_75

# if-target
circle if-target1.cxx -sm_75
circle enum1.cxx
circle enum2.cxx
circle arch.cxx -c -sm_35 -sm_52 -sm_61 -sm_75
circle if-target2.cxx -sm_35 -sm_52 -sm_61 -sm_75

# target overrides
circle override.cxx -sm_35 -sm_52 -sm_61 -sm_75
# then remove -sm_75 and show how different override is chosen when sm_61 is
# the best match
circle override.cxx -sm_35 -sm_52 -sm_61

# my device is sm_75. This lets me choose the best fit.
circle launch.cxx -sm_35 -sm_52 -sm_61 -sm_80