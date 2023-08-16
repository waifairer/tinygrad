#include <stdio.h>
#include <OpenCL/opencl.h>
#include <stdio.h>
#include <cassert>
#include <patch_list.h>
#include <vector>

#define CL_KERNEL_BINARY_PROGRAM_INTEL 0x407D

int main() {
    cl_int status;

    // Create a context
    cl_platform_id platform;
    status = clGetPlatformIDs(1, &platform, NULL);
    assert(status == CL_SUCCESS);

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    cl_context context = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    // Create command queue
    cl_device_id device;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    printf("Device ID: %p\n", device);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
    assert(status == CL_SUCCESS);

    // Create a program from the OpenCL source code
    const char* source_code =
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable \n"
        "__kernel void E_2_4(__global unsigned char* data0, const __global half* data1) { "
            "int gidx0 = get_group_id(0);  /* 2 */ "
            "float4 val1_0 = vload_half4(0, data1+gidx0*4); "
            "data0[(gidx0 * 4)] = val1_0.x; "
            "data0[(gidx0 * 4) + 1] = val1_0.y; "
            "data0[(gidx0 * 4) + 2] = val1_0.z; "
            "data0[(gidx0 * 4) + 3] = val1_0.w; "
        "} ";

    cl_program program = clCreateProgramWithSource(context, 1, &source_code, NULL, &status);
    assert(status == CL_SUCCESS);

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "E_2_4", &status);

    int device_count = 2;
    size_t* binary_size = new size_t[device_count];
    status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                            device_count * sizeof(size_t),
                            binary_size, nullptr);
    assert(status == CL_SUCCESS);

    uint8_t** binary = new uint8_t* [device_count];
    for (size_t i = 0; i < device_count; ++i) {
        binary[i] = new uint8_t[binary_size[i]];
    }

    status = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                            device_count * sizeof(uint8_t*),
                            binary, nullptr);
    assert(status == CL_SUCCESS);
    const iOpenCL::SProgramBinaryHeader* header =
    reinterpret_cast<const iOpenCL::SProgramBinaryHeader*>(binary[0]);
    printf("Binary: %p\n", binary[1]);
    printf("Magic: %x\n", header->Magic);
    printf("Version: %x\n", header->Version);
    printf("Device: %x\n", header->Device);
    printf("numkernels: %x\n", header->NumberOfKernels);
    

    assert(header->Magic == iOpenCL::MAGIC_CL);

    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(header) +
        sizeof(iOpenCL::SProgramBinaryHeader) + header->PatchListSize;
    for (uint32_t i = 0; i < header->NumberOfKernels; ++i) {
        const iOpenCL::SKernelBinaryHeaderCommon* kernel_header =
            reinterpret_cast<const iOpenCL::SKernelBinaryHeaderCommon*>(ptr);

        ptr += sizeof(iOpenCL::SKernelBinaryHeaderCommon);
        const char* kernel_name = reinterpret_cast<const char*>(ptr);

        ptr += kernel_header->KernelNameSize;
        if (strcmp(kernel_name, "E_2_4") == 0) {
            std::vector<uint8_t> raw_binary(kernel_header->KernelHeapSize);
            memcpy(raw_binary.data(), ptr,
                kernel_header->KernelHeapSize * sizeof(uint8_t));
        }

        ptr += kernel_header->PatchListSize +
            kernel_header->KernelHeapSize +
            kernel_header->GeneralStateHeapSize + kernel_header->DynamicStateHeapSize +
            kernel_header->SurfaceStateHeapSize;
    }
    unsigned char data0[4]; // output
    __fp16 data1[4] = {1.0f, 2.0f, 3.0f, 4.0f}; // input

    // Create memory buffers
    cl_mem buffer0 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * 4, data0, &status);
    cl_mem buffer1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, data1, &status);

    // Set kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer0);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer1);

    // Enqueue the kernel
    size_t global_work_size = 2; // Number of work-items
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

    // Read the result
    status = clEnqueueReadBuffer(queue, buffer0, CL_TRUE, 0, sizeof(unsigned char) * 4, data0, 0, NULL, NULL);

    // Output the result
    for (int i = 0; i < 4; i++) {
        printf("%u ", data0[i]);
    }
    printf("\n");

    // Clean up resources
    clReleaseMemObject(buffer0);
    clReleaseMemObject(buffer1);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}