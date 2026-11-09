#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

// External CRT & Windows functions
int printf(const char *fmt, ...);
int puts(const char *str);
void *malloc(size_t size);
void free(void *ptr);
void *memcpy(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);
void exit(int code);
void *fopen(const char *filename, const char *mode);
int fclose(void *stream);
size_t fread(void *ptr, size_t size, size_t nmemb, void *stream);
int fseek(void *stream, long offset, int whence);
long ftell(void *stream);

typedef void *HANDLE;
typedef int BOOL;
typedef union _LARGE_INTEGER {
  struct {
    uint32_t LowPart;
    int32_t HighPart;
  };
  int64_t QuadPart;
} LARGE_INTEGER;

__declspec(dllimport) BOOL __stdcall
QueryPerformanceCounter(LARGE_INTEGER *lpPerformanceCount);
__declspec(dllimport) BOOL __stdcall
QueryPerformanceFrequency(LARGE_INTEGER *lpFrequency);

int _fltused = 0x9875;

#define VK_CHECK(x)                                                            \
  do {                                                                         \
    VkResult err = (x);                                                        \
    if (err != VK_SUCCESS) {                                                   \
      printf("Vulkan error at line %d: code %d\n", __LINE__, (int)err);        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Read binary file
static uint32_t *readSpirvFile(const char *filename, size_t *outSize) {
  void *fp = fopen(filename, "rb");
  if (!fp) {
    printf("Error: Could not open SPIR-V file: %s\n", filename);
    exit(1);
  }
  fseek(fp, 0, 2); // SEEK_END
  long size = ftell(fp);
  fseek(fp, 0, 0); // SEEK_SET

  uint32_t *buffer = (uint32_t *)malloc(size);
  if (!buffer) {
    printf("Error: Memory allocation failed for SPIR-V code\n");
    exit(1);
  }
  size_t readBytes = fread(buffer, 1, size, fp);
  fclose(fp);
  *outSize = readBytes;
  return buffer;
}

// Find appropriate memory type
static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeBits,
                               VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeBits & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  // Fallback if exact match not found
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if (typeBits & (1 << i)) {
      return i;
    }
  }
  printf("Error: Suitable memory type not found!\n");
  exit(1);
}

// Create a GPU buffer and allocate memory
static void createBuffer(VkDevice device, VkPhysicalDevice physDev,
                         VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags props, VkBuffer *outBuf,
                         VkDeviceMemory *outMem) {
  VkBufferCreateInfo bufInfo = {0};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = size;
  bufInfo.usage = usage;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(device, &bufInfo, NULL, outBuf));

  VkMemoryRequirements memReq;
  vkGetBufferMemoryRequirements(device, *outBuf, &memReq);

  VkMemoryAllocateInfo allocInfo = {0};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReq.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(physDev, memReq.memoryTypeBits, props);
  VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, outMem));

  VK_CHECK(vkBindBufferMemory(device, *outBuf, *outMem, 0));
}

int mainCRTStartup() {
  puts("======================================================================="
       "=========");
  puts("                 GPU Memory Bandwidth Benchmark (Vulkan Compute)       "
       "         ");
  puts("======================================================================="
       "=========");

  // 1. Create Vulkan Instance
  VkApplicationInfo appInfo = {0};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "BandwidthBenchmark";
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo instInfo = {0};
  instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instInfo.pApplicationInfo = &appInfo;

  VkInstance instance;
  VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

  // 2. Select Physical Device
  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(instance, &devCount, NULL);
  if (devCount == 0) {
    puts("Error: No Vulkan physical device found!");
    exit(1);
  }
  VkPhysicalDevice *physDevices =
      (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * devCount);
  vkEnumeratePhysicalDevices(instance, &devCount, physDevices);
  VkPhysicalDevice physDev = physDevices[0];
  free(physDevices);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDev, &props);
  printf("GPU Device:        %s\n", props.deviceName);
  printf("Driver Version:    %u.%u.%u\n", props.driverVersion >> 22,
         (props.driverVersion >> 12) & 0x3ff, props.driverVersion & 0xfff);
  printf("Vulkan API:        %u.%u.%u\n", VK_VERSION_MAJOR(props.apiVersion),
         VK_VERSION_MINOR(props.apiVersion),
         VK_VERSION_PATCH(props.apiVersion));
  printf("Timestamp Period:  %.4f ns\n", props.limits.timestampPeriod);

  // Find compute queue
  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
  VkQueueFamilyProperties *qfProps = (VkQueueFamilyProperties *)malloc(
      sizeof(VkQueueFamilyProperties) * qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfProps);

  uint32_t computeQueueIdx = UINT32_MAX;
  for (uint32_t i = 0; i < qfCount; ++i) {
    if ((qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        (qfProps[i].timestampValidBits > 0)) {
      computeQueueIdx = i;
      break;
    }
  }
  free(qfProps);

  if (computeQueueIdx == UINT32_MAX) {
    puts("Error: No compute queue with timestamp support found!");
    exit(1);
  }

  // 3. Create Logical Device
  float qPriority = 1.0f;
  VkDeviceQueueCreateInfo qInfo = {0};
  qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qInfo.queueFamilyIndex = computeQueueIdx;
  qInfo.queueCount = 1;
  qInfo.pQueuePriorities = &qPriority;

  VkDeviceCreateInfo devInfo = {0};
  devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  devInfo.queueCreateInfoCount = 1;
  devInfo.pQueueCreateInfos = &qInfo;

  VkDevice device;
  VK_CHECK(vkCreateDevice(physDev, &devInfo, NULL, &device));

  VkQueue queue;
  vkGetDeviceQueue(device, computeQueueIdx, 0, &queue);

  // 4. Benchmark Configuration
  // 4 million vec4/uvec4 elements = 64 MB per buffer
  // Large enough to completely surpass L1/L2/L3 caches and measure real memory
  // bus bandwidth.
  const uint32_t numElements = 4 * 1024 * 1024;
  const VkDeviceSize bufferSizeBytes =
      (VkDeviceSize)numElements * sizeof(uint32_t) * 4; // 64 MB
  const uint32_t unrollFactor = 4;
  const uint32_t workgroupSize = 256;
  const uint32_t elementsPerGroup = workgroupSize * unrollFactor; // 1024
  const uint32_t numGroups =
      (numElements + elementsPerGroup - 1) / elementsPerGroup; // 4096 groups

  printf("Buffer Size:       %.2f MB per buffer\n",
         (double)bufferSizeBytes / (1024.0 * 1024.0));
  printf("Total Elements:    %u (128-bit vec4s)\n", numElements);
  printf("Workgroups:        %u (%u threads/group, 4x unrolled)\n", numGroups,
         workgroupSize);
  puts("-----------------------------------------------------------------------"
       "---------");

  // 5. Create Storage Buffers (A, B, C)
  VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkBuffer bufA, bufB, bufC;
  VkDeviceMemory memA, memB, memC;

  createBuffer(device, physDev, bufferSizeBytes,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufA, &memA);
  createBuffer(device, physDev, bufferSizeBytes,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufB, &memB);
  createBuffer(device, physDev, bufferSizeBytes,
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memFlags, &bufC, &memC);

  // Initialize Input Buffers A and B
  float *ptrA;
  float *ptrB;
  VK_CHECK(vkMapMemory(device, memA, 0, bufferSizeBytes, 0, (void **)&ptrA));
  VK_CHECK(vkMapMemory(device, memB, 0, bufferSizeBytes, 0, (void **)&ptrB));
  for (uint32_t i = 0; i < numElements * 4; ++i) {
    ptrA[i] = 1.0f;
    ptrB[i] = 2.0f;
  }
  vkUnmapMemory(device, memA);
  vkUnmapMemory(device, memB);

  // 6. Query Pool for Hardware GPU Timestamps
  VkQueryPoolCreateInfo qpInfo = {0};
  qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpInfo.queryCount = 2; // start & end

  VkQueryPool queryPool;
  VK_CHECK(vkCreateQueryPool(device, &qpInfo, NULL, &queryPool));

  // 7. Command Pool & Command Buffer
  VkCommandPoolCreateInfo cpInfo = {0};
  cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpInfo.queueFamilyIndex = computeQueueIdx;
  cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  VkCommandPool cmdPool;
  VK_CHECK(vkCreateCommandPool(device, &cpInfo, NULL, &cmdPool));

  VkCommandBufferAllocateInfo cbAlloc = {0};
  cbAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAlloc.commandPool = cmdPool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1;

  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(device, &cbAlloc, &cmd));

  // 8. Descriptor Pool
  VkDescriptorPoolSize poolSizes[1];
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[0].descriptorCount = 10;

  VkDescriptorPoolCreateInfo dpInfo = {0};
  dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpInfo.maxSets = 4;
  dpInfo.poolSizeCount = 1;
  dpInfo.pPoolSizes = poolSizes;

  VkDescriptorPool descPool;
  VK_CHECK(vkCreateDescriptorPool(device, &dpInfo, NULL, &descPool));

  // ========================================================================
  // TEST 1: COPY BENCHMARK (C = A) -> 1 Read + 1 Write = 2x BufferSize
  // ========================================================================
  {
    puts("[Running Benchmark 1/2: GPU Copy Throughput]");

    // Descriptor Set Layout for Copy (binding 0: in, binding 1: out)
    VkDescriptorSetLayoutBinding copyBindings[2] = {0};
    copyBindings[0].binding = 0;
    copyBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    copyBindings[0].descriptorCount = 1;
    copyBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    copyBindings[1].binding = 1;
    copyBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    copyBindings[1].descriptorCount = 1;
    copyBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo copyDslInfo = {0};
    copyDslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    copyDslInfo.bindingCount = 2;
    copyDslInfo.pBindings = copyBindings;

    VkDescriptorSetLayout copyDsl;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &copyDslInfo, NULL, &copyDsl));

    VkPushConstantRange pcRange = {0};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo copyPlInfo = {0};
    copyPlInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    copyPlInfo.setLayoutCount = 1;
    copyPlInfo.pSetLayouts = &copyDsl;
    copyPlInfo.pushConstantRangeCount = 1;
    copyPlInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout copyPipelineLayout;
    VK_CHECK(
        vkCreatePipelineLayout(device, &copyPlInfo, NULL, &copyPipelineLayout));

    // Load SPIR-V Shader Module
    size_t spvSize = 0;
    uint32_t *spvCode = readSpirvFile("spirv/copy_bandwidth.spv", &spvSize);

    VkShaderModuleCreateInfo smInfo = {0};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = spvSize;
    smInfo.pCode = spvCode;

    VkShaderModule copyModule;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &copyModule));
    free(spvCode);

    // Compute Pipeline
    VkComputePipelineCreateInfo cpPipeInfo = {0};
    cpPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpPipeInfo.layout = copyPipelineLayout;
    cpPipeInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpPipeInfo.stage.module = copyModule;
    cpPipeInfo.stage.pName = "main";

    VkPipeline copyPipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpPipeInfo,
                                      NULL, &copyPipeline));

    // Descriptor Set
    VkDescriptorSetAllocateInfo dsAlloc = {0};
    dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool = descPool;
    dsAlloc.descriptorSetCount = 1;
    dsAlloc.pSetLayouts = &copyDsl;

    VkDescriptorSet copyDescSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsAlloc, &copyDescSet));

    VkDescriptorBufferInfo bufInfoA = {bufA, 0, bufferSizeBytes};
    VkDescriptorBufferInfo bufInfoC = {bufC, 0, bufferSizeBytes};

    VkWriteDescriptorSet copyWrites[2] = {0};
    copyWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    copyWrites[0].dstSet = copyDescSet;
    copyWrites[0].dstBinding = 0;
    copyWrites[0].descriptorCount = 1;
    copyWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    copyWrites[0].pBufferInfo = &bufInfoA;

    copyWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    copyWrites[1].dstSet = copyDescSet;
    copyWrites[1].dstBinding = 1;
    copyWrites[1].descriptorCount = 1;
    copyWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    copyWrites[1].pBufferInfo = &bufInfoC;

    vkUpdateDescriptorSets(device, 2, copyWrites, 0, NULL);

    // Record Command Buffer
    VkCommandBufferBeginInfo cbBegin = {0};
    cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));
    vkCmdResetQueryPool(cmd, queryPool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, copyPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            copyPipelineLayout, 0, 1, &copyDescSet, 0, NULL);
    vkCmdPushConstants(cmd, copyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(uint32_t), &numElements);
    vkCmdDispatch(cmd, numGroups, 1, 1);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool,
                        1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Warmup runs (3 passes)
    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    for (int w = 0; w < 3; ++w) {
      VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));
    }

    // Benchmark Runs (10 iterations)
    const int iterations = 10;
    double bestGpuTimeMs = 1e9;
    double totalGpuTimeMs = 0;
    uint64_t timestamps[2] = {0};

    // 1 Read + 1 Write = 2x BufferSize
    double totalBytesTransferred = 2.0 * (double)bufferSizeBytes;

    for (int iter = 0; iter < iterations; ++iter) {
      VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));

      VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, 2,
                                     sizeof(timestamps), timestamps,
                                     sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
      double timeNs = (double)(timestamps[1] - timestamps[0]) *
                      (double)props.limits.timestampPeriod;
      double timeMs = timeNs / 1.0e6;

      if (timeMs < bestGpuTimeMs)
        bestGpuTimeMs = timeMs;
      totalGpuTimeMs += timeMs;
    }

    double avgGpuTimeMs = totalGpuTimeMs / iterations;
    double peakBandwidthGBs =
        (totalBytesTransferred / (bestGpuTimeMs / 1000.0)) / 1.0e9;
    double avgBandwidthGBs =
        (totalBytesTransferred / (avgGpuTimeMs / 1000.0)) / 1.0e9;

    // Verify data
    uint32_t *ptrC;
    VK_CHECK(vkMapMemory(device, memC, 0, bufferSizeBytes, 0, (void **)&ptrC));
    int verified = 1;
    for (uint32_t i = 0; i < 1000; ++i) {
      if (ptrC[i] != ((uint32_t *)ptrA)[i]) {
        verified = 0;
        break;
      }
    }
    vkUnmapMemory(device, memC);

    printf("  Bytes Transferred:   %.2f MB (Read A + Write C)\n",
           totalBytesTransferred / (1024.0 * 1024.0));
    printf("  Best GPU Kernel:     %.3f ms\n", bestGpuTimeMs);
    printf("  Average GPU Kernel:  %.3f ms\n", avgGpuTimeMs);
    printf("  >> REAL PEAK COPY BANDWIDTH:    %.2f GB/s\n", peakBandwidthGBs);
    printf("  >> AVERAGE COPY BANDWIDTH:      %.2f GB/s\n", avgBandwidthGBs);
    printf("  Verification:        %s\n",
           verified ? "PASSED (Data matches exactly)" : "FAILED");
    puts("---------------------------------------------------------------------"
         "-----------");

    vkDestroyPipeline(device, copyPipeline, NULL);
    vkDestroyPipelineLayout(device, copyPipelineLayout, NULL);
    vkDestroyShaderModule(device, copyModule, NULL);
    vkDestroyDescriptorSetLayout(device, copyDsl, NULL);
  }

  // ========================================================================
  // TEST 2: VECTOR ADD BENCHMARK (C = A + B) -> 2 Reads + 1 Write = 3x
  // BufferSize
  // ========================================================================
  {
    puts("[Running Benchmark 2/2: GPU Vector Add (A + B = C) Throughput]");

    // Descriptor Set Layout for Vector Add (bindings 0, 1, 2)
    VkDescriptorSetLayoutBinding addBindings[3] = {0};
    for (int i = 0; i < 3; ++i) {
      addBindings[i].binding = i;
      addBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      addBindings[i].descriptorCount = 1;
      addBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo addDslInfo = {0};
    addDslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    addDslInfo.bindingCount = 3;
    addDslInfo.pBindings = addBindings;

    VkDescriptorSetLayout addDsl;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &addDslInfo, NULL, &addDsl));

    VkPushConstantRange pcRange = {0};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo addPlInfo = {0};
    addPlInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    addPlInfo.setLayoutCount = 1;
    addPlInfo.pSetLayouts = &addDsl;
    addPlInfo.pushConstantRangeCount = 1;
    addPlInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout addPipelineLayout;
    VK_CHECK(
        vkCreatePipelineLayout(device, &addPlInfo, NULL, &addPipelineLayout));

    // Load SPIR-V Shader Module
    size_t spvSize = 0;
    uint32_t *spvCode =
        readSpirvFile("spirv/vector_add_bandwidth.spv", &spvSize);

    VkShaderModuleCreateInfo smInfo = {0};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = spvSize;
    smInfo.pCode = spvCode;

    VkShaderModule addModule;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &addModule));
    free(spvCode);

    // Compute Pipeline
    VkComputePipelineCreateInfo cpPipeInfo = {0};
    cpPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpPipeInfo.layout = addPipelineLayout;
    cpPipeInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpPipeInfo.stage.module = addModule;
    cpPipeInfo.stage.pName = "main";

    VkPipeline addPipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpPipeInfo,
                                      NULL, &addPipeline));

    // Descriptor Set
    VkDescriptorSetAllocateInfo dsAlloc = {0};
    dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool = descPool;
    dsAlloc.descriptorSetCount = 1;
    dsAlloc.pSetLayouts = &addDsl;

    VkDescriptorSet addDescSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsAlloc, &addDescSet));

    VkDescriptorBufferInfo bufInfoA = {bufA, 0, bufferSizeBytes};
    VkDescriptorBufferInfo bufInfoB = {bufB, 0, bufferSizeBytes};
    VkDescriptorBufferInfo bufInfoC = {bufC, 0, bufferSizeBytes};

    VkWriteDescriptorSet addWrites[3] = {0};
    addWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    addWrites[0].dstSet = addDescSet;
    addWrites[0].dstBinding = 0;
    addWrites[0].descriptorCount = 1;
    addWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    addWrites[0].pBufferInfo = &bufInfoA;

    addWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    addWrites[1].dstSet = addDescSet;
    addWrites[1].dstBinding = 1;
    addWrites[1].descriptorCount = 1;
    addWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    addWrites[1].pBufferInfo = &bufInfoB;

    addWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    addWrites[2].dstSet = addDescSet;
    addWrites[2].dstBinding = 2;
    addWrites[2].descriptorCount = 1;
    addWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    addWrites[2].pBufferInfo = &bufInfoC;

    vkUpdateDescriptorSets(device, 3, addWrites, 0, NULL);

    // Record Command Buffer
    VkCommandBufferBeginInfo cbBegin = {0};
    cbBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VK_CHECK(vkBeginCommandBuffer(cmd, &cbBegin));
    vkCmdResetQueryPool(cmd, queryPool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, addPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            addPipelineLayout, 0, 1, &addDescSet, 0, NULL);
    vkCmdPushConstants(cmd, addPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(uint32_t), &numElements);
    vkCmdDispatch(cmd, numGroups, 1, 1);

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool,
                        1);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // Warmup (3 passes)
    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    for (int w = 0; w < 3; ++w) {
      VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));
    }

    // Benchmark Runs (10 iterations)
    const int iterations = 10;
    double bestGpuTimeMs = 1e9;
    double totalGpuTimeMs = 0;
    uint64_t timestamps[2] = {0};

    // 2 Reads + 1 Write = 3x BufferSize
    double totalBytesTransferred = 3.0 * (double)bufferSizeBytes;

    for (int iter = 0; iter < iterations; ++iter) {
      VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
      VK_CHECK(vkQueueWaitIdle(queue));

      VK_CHECK(vkGetQueryPoolResults(device, queryPool, 0, 2,
                                     sizeof(timestamps), timestamps,
                                     sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
      double timeNs = (double)(timestamps[1] - timestamps[0]) *
                      (double)props.limits.timestampPeriod;
      double timeMs = timeNs / 1.0e6;

      if (timeMs < bestGpuTimeMs)
        bestGpuTimeMs = timeMs;
      totalGpuTimeMs += timeMs;
    }

    double avgGpuTimeMs = totalGpuTimeMs / iterations;
    double peakBandwidthGBs =
        (totalBytesTransferred / (bestGpuTimeMs / 1000.0)) / 1.0e9;
    double avgBandwidthGBs =
        (totalBytesTransferred / (avgGpuTimeMs / 1000.0)) / 1.0e9;

    // Verify data (1.0 + 2.0 = 3.0)
    float *ptrC;
    VK_CHECK(vkMapMemory(device, memC, 0, bufferSizeBytes, 0, (void **)&ptrC));
    int verified = 1;
    for (uint32_t i = 0; i < 1000; ++i) {
      if (ptrC[i] != 3.0f) {
        verified = 0;
        break;
      }
    }
    vkUnmapMemory(device, memC);

    printf("  Bytes Transferred:   %.2f MB (Read A + Read B + Write C)\n",
           totalBytesTransferred / (1024.0 * 1024.0));
    printf("  Best GPU Kernel:     %.3f ms\n", bestGpuTimeMs);
    printf("  Average GPU Kernel:  %.3f ms\n", avgGpuTimeMs);
    printf("  >> REAL PEAK VECTOR-ADD BANDWIDTH: %.2f GB/s\n",
           peakBandwidthGBs);
    printf("  >> AVERAGE VECTOR-ADD BANDWIDTH:   %.2f GB/s\n", avgBandwidthGBs);
    printf("  Verification:        %s\n",
           verified ? "PASSED (1.0 + 2.0 = 3.0)" : "FAILED");
    puts("====================================================================="
         "===========");

    vkDestroyPipeline(device, addPipeline, NULL);
    vkDestroyPipelineLayout(device, addPipelineLayout, NULL);
    vkDestroyShaderModule(device, addModule, NULL);
    vkDestroyDescriptorSetLayout(device, addDsl, NULL);
  }

  // Cleanup resources
  vkDestroyDescriptorPool(device, descPool, NULL);
  vkDestroyCommandPool(device, cmdPool, NULL);
  vkDestroyQueryPool(device, queryPool, NULL);

  vkDestroyBuffer(device, bufA, NULL);
  vkFreeMemory(device, memA, NULL);
  vkDestroyBuffer(device, bufB, NULL);
  vkFreeMemory(device, memB, NULL);
  vkDestroyBuffer(device, bufC, NULL);
  vkFreeMemory(device, memC, NULL);

  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);

  puts("Benchmark completed cleanly!");
  exit(0);
}
