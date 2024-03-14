#include <OpenCL/OpenCL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 内嵌的 OpenCL 内核源代码 - 标量版本
const char* kernelSourceScalar =
"__kernel void vector_add_scalar(__global const float* A, __global const float* B, __global float* C, const int numElements) {\n"
"    int i = get_global_id(0);\n"
"    if (i < numElements) {\n"
"        C[i] = A[i] + B[i];\n"
"    }\n"
"}\n";

// 内嵌的 OpenCL 内核源代码 - 向量化版本
const char* kernelSourceVectorized =
"__kernel void vector_add_vectorized(__global const float4* A, __global const float4* B, __global float4* C, const int numElements) {\n"
"    int i = get_global_id(0);\n"
"    if (i < numElements) {\n"
"        C[i] = A[i] + B[i];\n"
"    }\n"
"}\n";


void testVectorAdd(const char* kernelSource, const char* kernelName, size_t globalWorkSize, size_t localWorkSize, const char* testDescription) {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufA, bufB, bufC;
    float *A, *B, *C;
    size_t numElements = globalWorkSize;
    cl_int err;
    
    // 获取第一个平台和设备
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 创建上下文和命令队列
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    // 从源代码创建程序对象
    const char* sources[] = { kernelSource };
    program = clCreateProgramWithSource(context, 1, sources, NULL, &err);

    // 编译程序
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // 如果编译失败，打印错误信息
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        printf("Error in kernel: %s\n", buildLog);
        exit(-1);
    }

    // 创建内核对象
    kernel = clCreateKernel(program, kernelName, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create kernel: %d\n", err);
        exit(-1);
    }

    // 分配并初始化输入和输出数据
    A = (float*)malloc(numElements * sizeof(float));
    B = (float*)malloc(numElements * sizeof(float));
    C = (float*)malloc(numElements * sizeof(float));
    for (int i = 0; i < numElements; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    // 创建缓冲对象
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, numElements * sizeof(float), NULL, &err);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, numElements * sizeof(float), NULL, &err);
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numElements * sizeof(float), NULL, &err);

    // 写入数据到缓冲区
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, numElements * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, numElements * sizeof(float), B, 0, NULL, NULL);

    // 设置内核参数
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &numElements);
    if (err != CL_SUCCESS) {
        printf("Failed to set kernel arguments: %d\n", err);
        exit(-1);
    }

    // 执行内核
    cl_event event;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("Failed to enqueue NDRange kernel: %d\n", err);
        exit(-1);
    }
    
    // 等待事件完成
    clWaitForEvents(1, &event);
    if (err != CL_SUCCESS) {
        printf("Failed to wait for event completion: %d\n", err);
        exit(-1);
    }

    // 计算并输出执行时间
    cl_ulong timeStart, timeEnd;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);
    double totalTime = (double)(timeEnd - timeStart) * 1e-6; // 转换为毫秒
    printf("%s - Execution time (globalWorkSize=%zu, localWorkSize=%zu): %f ms\n", testDescription, globalWorkSize, localWorkSize, totalTime);

    // 释放资源
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C);
}

int main() {
    // 分别对标量和向量化版本进行测试，说明每次测试的配置
    testVectorAdd(kernelSourceScalar, "vector_add_scalar", 9600000, 160, "Scalar version with localWorkSize=160");
    testVectorAdd(kernelSourceScalar, "vector_add_scalar", 9600000, 256, "Scalar version with localWorkSize=256");
    
    // 对向量化版本的测试需要调整全局工作大小，因为每个工作项现在处理4个元素
    size_t globalWorkSizeVectorized = 9600000 / 4;
    testVectorAdd(kernelSourceVectorized, "vector_add_vectorized", globalWorkSizeVectorized, 40, "Vectorized version with localWorkSize=40");
    testVectorAdd(kernelSourceVectorized, "vector_add_vectorized", globalWorkSizeVectorized, 64, "Vectorized version with localWorkSize=64");

    return 0;
}
