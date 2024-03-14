#include <OpenCL/opencl.h>
#include <stdio.h>
#include <stdlib.h>

void printPlatformInfo(cl_platform_id platform) {
    char platform_name[1024];
    char platform_vendor[1024];
    char platform_version[1024];

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);

    printf("Platform Name: %s\n", platform_name);
    printf("Platform Vendor: %s\n", platform_vendor);
    printf("Platform Version: %s\n", platform_version);
}

void printDeviceInfo(cl_device_id device) {
    // Device identification information
    char device_name[1024];
    char device_vendor[1024];
    char device_version[1024];
    cl_device_type device_type;
    cl_bool device_available;
    
    // Compute resources
    cl_uint max_compute_units;
    cl_uint max_clock_frequency;
    size_t max_work_group_size;
    cl_uint max_work_item_dimensions;
    size_t max_work_item_sizes[3];
    
    // Memory information
    cl_ulong global_mem_size;
    cl_ulong max_mem_alloc_size;
    cl_ulong local_mem_size;
    cl_ulong max_parameter_size;
    cl_ulong global_mem_cache_size;
    cl_uint global_mem_cacheline_size;
    cl_bool error_correction_support;
    
    // Performance and capabilities
    cl_device_exec_capabilities exec_capabilities;
    size_t profiling_timer_resolution;
    cl_bool endian_little;
    cl_bool compiler_available;
    char device_extensions[2048];
    
    // Vector widths
    cl_uint preferred_vector_width_char;
    cl_uint preferred_vector_width_short;
    cl_uint preferred_vector_width_int;
    cl_uint preferred_vector_width_long;
    cl_uint preferred_vector_width_float;
    cl_uint preferred_vector_width_double;
    
    // Retrieve and print device information
    // Note: Error checking is omitted for brevity, but should be included in production code

    // Device identification
    printf("Device Identification:\n");
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    clGetDeviceInfo(device, CL_DEVICE_AVAILABLE, sizeof(device_available), &device_available, NULL);
    printf("\tName: %s\n", device_name);
    printf("\tVendor: %s\n", device_vendor);
    printf("\tVersion: %s\n", device_version);
    printf("\tAvailable: %s\n", device_available ? "Yes" : "No");
    // Device type output omitted for brevity

    // Compute resources
    printf("Compute Resources:\n");
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency), &max_clock_frequency, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes, NULL);
    printf("\tMax Compute Units: %u\n", max_compute_units);
    printf("\tMax Clock Frequency: %u MHz\n", max_clock_frequency);
    printf("\tMax Work Group Size: %zu\n", max_work_group_size);
    printf("\tMax Work Item Dimensions: %u\n", max_work_item_dimensions);
    printf("\tMax Work Item Sizes: %zu, %zu, %zu\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);

    // Memory information
    printf("Memory Information:\n");
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(global_mem_cache_size), &global_mem_cache_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(global_mem_cacheline_size), &global_mem_cacheline_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
    printf("\tGlobal Memory Size: %llu bytes\n", (unsigned long long)global_mem_size);
    printf("\tMax Memory Allocation Size: %llu bytes\n", (unsigned long long)max_mem_alloc_size);
    printf("\tLocal Memory Size: %llu bytes\n", (unsigned long long)local_mem_size);
    printf("\tGlobal Memory Cache Size: %llu bytes\n", (unsigned long long)global_mem_cache_size);
    printf("\tGlobal Memory Cache Line Size: %u bytes\n", global_mem_cacheline_size);
    printf("\tError Correction Support: %s\n", error_correction_support ? "Yes" : "No");

    // Performance and capabilities
    printf("Performance and Capabilities:\n");
    clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(profiling_timer_resolution), &profiling_timer_resolution, NULL);
    clGetDeviceInfo(device, CL_DEVICE_ENDIAN_LITTLE, sizeof(endian_little), &endian_little, NULL);
    clGetDeviceInfo(device, CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available), &compiler_available, NULL);
    clGetDeviceInfo(device, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(exec_capabilities), &exec_capabilities, NULL);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(device_extensions), device_extensions, NULL);
    printf("\tProfiling Timer Resolution: %zu nanoseconds\n", profiling_timer_resolution);
    printf("\tLittle Endian: %s\n", endian_little ? "Yes" : "No");
    printf("\tCompiler Available: %s\n", compiler_available ? "Yes" : "No");
    printf("\tExecution Capabilities: ");
    if (exec_capabilities & CL_EXEC_KERNEL) {
        printf("OpenCL Kernels ");
    }
    if (exec_capabilities & CL_EXEC_NATIVE_KERNEL) {
        printf("Native Kernels ");
    }
    printf("\n\tExtensions: %s\n", device_extensions);

    // Vector widths
    printf("Preferred Vector Widths:\n");
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(preferred_vector_width_char), &preferred_vector_width_char, NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(preferred_vector_width_short), &preferred_vector_width_short, NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(preferred_vector_width_int), &preferred_vector_width_int, NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(preferred_vector_width_long), &preferred_vector_width_long, NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(preferred_vector_width_float), &preferred_vector_width_float, NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(preferred_vector_width_double), &preferred_vector_width_double, NULL);
    printf("\tChar: %u\n", preferred_vector_width_char);
    printf("\tShort: %u\n", preferred_vector_width_short);
    printf("\tInt: %u\n", preferred_vector_width_int);
    printf("\tLong: %u\n", preferred_vector_width_long);
    printf("\tFloat: %u\n", preferred_vector_width_float);
    printf("\tDouble: %u\n", preferred_vector_width_double);
}

int main() {
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    printf("Number of platforms: %u\n", num_platforms);

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    for (cl_uint i = 0; i < num_platforms; i++) {
        printf("Platform #%u:\n", i);
        printPlatformInfo(platforms[i]);

        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        printf("\tNumber of devices: %u\n", num_devices);
        for (cl_uint j = 0; j < num_devices; j++) {
            printDeviceInfo(devices[j]);
        }
        free(devices);
    }

    free(platforms);
    return 0;
}
