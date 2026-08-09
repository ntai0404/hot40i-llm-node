#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint32_t VkFlags;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef int32_t VkResult;
typedef struct VkInstance_T* VkInstance;
typedef struct VkPhysicalDevice_T* VkPhysicalDevice;

#define VK_SUCCESS 0
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1

typedef struct VkApplicationInfo {
  uint32_t sType;
  const void* pNext;
  const char* pApplicationName;
  uint32_t applicationVersion;
  const char* pEngineName;
  uint32_t engineVersion;
  uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
  uint32_t sType;
  const void* pNext;
  VkFlags flags;
  const VkApplicationInfo* pApplicationInfo;
  uint32_t enabledLayerCount;
  const char* const* ppEnabledLayerNames;
  uint32_t enabledExtensionCount;
  const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkPhysicalDeviceLimits {
  uint32_t unused[256];
} VkPhysicalDeviceLimits;

typedef struct VkPhysicalDeviceSparseProperties {
  VkBool32 unused[5];
} VkPhysicalDeviceSparseProperties;

typedef struct VkPhysicalDeviceProperties {
  uint32_t apiVersion;
  uint32_t driverVersion;
  uint32_t vendorID;
  uint32_t deviceID;
  uint32_t deviceType;
  char deviceName[256];
  uint8_t pipelineCacheUUID[16];
  VkPhysicalDeviceLimits limits;
  VkPhysicalDeviceSparseProperties sparseProperties;
} VkPhysicalDeviceProperties;

typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*);
typedef void (*PFN_vkDestroyInstance)(VkInstance, const void*);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
typedef void (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*);

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_device_type;
typedef intptr_t cl_platform_id;
typedef intptr_t cl_device_id;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFFu
#define CL_PLATFORM_NAME 0x0902
#define CL_PLATFORM_VERSION 0x0901
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VERSION 0x102F

typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*PFN_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (*PFN_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);

static void print_json_string(const char* value) {
  putchar('"');
  for (const char* p = value; p && *p; ++p) {
    if (*p == '"' || *p == '\\') {
      putchar('\\');
    }
    putchar(*p);
  }
  putchar('"');
}

static void probe_vulkan() {
  void* lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  printf("\"vulkan\":{\"loadable\":%s", lib ? "true" : "false");
  if (!lib) {
    printf(",\"error\":");
    print_json_string(dlerror());
    printf("}");
    return;
  }
  PFN_vkCreateInstance vkCreateInstance = (PFN_vkCreateInstance)dlsym(lib, "vkCreateInstance");
  PFN_vkDestroyInstance vkDestroyInstance = (PFN_vkDestroyInstance)dlsym(lib, "vkDestroyInstance");
  PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices =
      (PFN_vkEnumeratePhysicalDevices)dlsym(lib, "vkEnumeratePhysicalDevices");
  PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties =
      (PFN_vkGetPhysicalDeviceProperties)dlsym(lib, "vkGetPhysicalDeviceProperties");
  if (!vkCreateInstance || !vkDestroyInstance || !vkEnumeratePhysicalDevices ||
      !vkGetPhysicalDeviceProperties) {
    printf(",\"error\":\"missing required Vulkan symbols\"}");
    dlclose(lib);
    return;
  }
  VkApplicationInfo app = {};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "hot40i-mlc-probe";
  app.apiVersion = (1u << 22) | (1u << 12);
  VkInstanceCreateInfo create = {};
  create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create.pApplicationInfo = &app;
  VkInstance instance = nullptr;
  VkResult rc = vkCreateInstance(&create, nullptr, &instance);
  printf(",\"create_instance_rc\":%d", rc);
  if (rc != VK_SUCCESS) {
    printf("}");
    dlclose(lib);
    return;
  }
  uint32_t count = 0;
  rc = vkEnumeratePhysicalDevices(instance, &count, nullptr);
  printf(",\"enumerate_rc\":%d,\"device_count\":%u,\"devices\":[", rc, count);
  if (rc == VK_SUCCESS && count > 0) {
    VkPhysicalDevice devices[8];
    uint32_t capped = count > 8 ? 8 : count;
    rc = vkEnumeratePhysicalDevices(instance, &capped, devices);
    for (uint32_t i = 0; i < capped; ++i) {
      VkPhysicalDeviceProperties props = {};
      vkGetPhysicalDeviceProperties(devices[i], &props);
      if (i) {
        putchar(',');
      }
      printf("{\"name\":");
      print_json_string(props.deviceName);
      printf(",\"api_version\":%u,\"vendor_id\":%u,\"device_id\":%u,\"device_type\":%u}",
             props.apiVersion, props.vendorID, props.deviceID, props.deviceType);
    }
  }
  printf("]}");
  vkDestroyInstance(instance, nullptr);
  dlclose(lib);
}

static void probe_opencl() {
  void* lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
  printf("\"opencl\":{\"loadable\":%s", lib ? "true" : "false");
  if (!lib) {
    printf(",\"error\":");
    print_json_string(dlerror());
    printf("}");
    return;
  }
  PFN_clGetPlatformIDs clGetPlatformIDs = (PFN_clGetPlatformIDs)dlsym(lib, "clGetPlatformIDs");
  PFN_clGetPlatformInfo clGetPlatformInfo = (PFN_clGetPlatformInfo)dlsym(lib, "clGetPlatformInfo");
  PFN_clGetDeviceIDs clGetDeviceIDs = (PFN_clGetDeviceIDs)dlsym(lib, "clGetDeviceIDs");
  PFN_clGetDeviceInfo clGetDeviceInfo = (PFN_clGetDeviceInfo)dlsym(lib, "clGetDeviceInfo");
  if (!clGetPlatformIDs || !clGetPlatformInfo || !clGetDeviceIDs || !clGetDeviceInfo) {
    printf(",\"error\":\"missing required OpenCL symbols\"}");
    dlclose(lib);
    return;
  }
  cl_uint platform_count = 0;
  cl_int rc = clGetPlatformIDs(0, nullptr, &platform_count);
  printf(",\"platform_rc\":%d,\"platform_count\":%u,\"platforms\":[", rc, platform_count);
  if (rc == CL_SUCCESS && platform_count > 0) {
    cl_platform_id platforms[8];
    cl_uint capped = platform_count > 8 ? 8 : platform_count;
    rc = clGetPlatformIDs(capped, platforms, nullptr);
    for (cl_uint i = 0; i < capped; ++i) {
      char platform_name[256] = {};
      char platform_version[256] = {};
      clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
      clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, nullptr);
      cl_uint device_count = 0;
      cl_int device_rc = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
      if (i) {
        putchar(',');
      }
      printf("{\"name\":");
      print_json_string(platform_name);
      printf(",\"version\":");
      print_json_string(platform_version);
      printf(",\"device_rc\":%d,\"device_count\":%u,\"devices\":[", device_rc, device_count);
      if (device_rc == CL_SUCCESS && device_count > 0) {
        cl_device_id devices[8];
        cl_uint capped_devices = device_count > 8 ? 8 : device_count;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, capped_devices, devices, nullptr);
        for (cl_uint j = 0; j < capped_devices; ++j) {
          char device_name[256] = {};
          char device_version[256] = {};
          clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
          clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(device_version), device_version, nullptr);
          if (j) {
            putchar(',');
          }
          printf("{\"name\":");
          print_json_string(device_name);
          printf(",\"version\":");
          print_json_string(device_version);
          printf("}");
        }
      }
      printf("]}");
    }
  }
  printf("]}");
  dlclose(lib);
}

int main() {
  printf("{");
  probe_vulkan();
  printf(",");
  probe_opencl();
  printf("}\n");
  return 0;
}
