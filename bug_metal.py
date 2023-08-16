import os
os.environ["METAL"] = "1"
import numpy as np
from tinygrad.helpers import dtypes
from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram

na = np.array([1,2,3,4], dtype=np.float16)
print(f"na = {na}")
a = RawMetalBuffer.fromCPU(na)
b = RawMetalBuffer(4, dtypes.uint8)
nb = b.toCPU().reshape(4)
print(f"nb = {nb}")

prog = MetalProgram("test", f"""
#include <metal_stdlib>
using namespace metal;
kernel void test(device char* data0, const device half4* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {{
  half4 val1_0 = *data1;
  data0[0] = val1_0.x;
  data0[1] = val1_0.y;
  data0[2] = val1_0.z;
  data0[3] = val1_0.w;
}}
""")
prog([1, 1, 1], [1, 1, 1], b, a, wait=True)

nb = b.toCPU().reshape(4)
print(f"nb'= {nb}") 