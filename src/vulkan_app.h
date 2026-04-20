#pragma once
#define GLFW_INCLUDE_VULKAN
#include "bvh.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

// GPU structs (must match shader std140/std430 layouts exactly)
struct alignas(16) Camera {
  glm::vec3 origin;
  float _p0;
  glm::vec3 lower_left;
  float _p1;
  glm::vec3 horizontal;
  float _p2;
  glm::vec3 vertical;
  float _p3;
  uint32_t frame_count;
  uint32_t width;
  uint32_t height;
  float _p4;
}; // 80 bytes

// structs from rtk files
struct RTKCamera {
  int fov;
  glm::vec3 position;
  glm::vec3 direction;
  bool active;
};

struct RTKObj {
  std::string objPath;
  std::string rtlPath;
  glm::vec3 position;
  glm::vec3 rotation;
  glm::vec3 scale;
};

struct RTKScene {
  std::vector<RTKObj> objects;
};

class VulkanApp {
public:
#ifdef __APPLE__
  static constexpr uint32_t W = 1280 * 3;
  static constexpr uint32_t H = 720 * 3;
#else
  static constexpr uint32_t W = 1280;
  static constexpr uint32_t H = 720;
#endif
  static constexpr int MAX_FRAMES = 2;

  void run();

private:
  // Window
  GLFWwindow *window = nullptr;

  // Core Vulkan
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice phys_dev = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue gfx_queue = VK_NULL_HANDLE;
  uint32_t gfx_family = 0;

  // Swapchain
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat sc_format = VK_FORMAT_UNDEFINED;
  VkExtent2D sc_extent = {};
  std::vector<VkImage> sc_images;
  std::vector<VkImageView> sc_views;

  // Render images
  VkImage accum_image = VK_NULL_HANDLE;
  VkDeviceMemory accum_mem = VK_NULL_HANDLE;
  VkImageView accum_view = VK_NULL_HANDLE;
  VkImage display_image = VK_NULL_HANDLE;
  VkDeviceMemory display_mem = VK_NULL_HANDLE;
  VkImageView display_view = VK_NULL_HANDLE;
  VkSampler display_sampler = VK_NULL_HANDLE;

  // Buffers
  VkBuffer camera_buf = VK_NULL_HANDLE;
  VkDeviceMemory camera_mem = VK_NULL_HANDLE;
  void *camera_map = nullptr;
  VkBuffer sphere_buf = VK_NULL_HANDLE;
  VkDeviceMemory sphere_mem = VK_NULL_HANDLE;
  VkBuffer bvh_buf = VK_NULL_HANDLE;
  VkDeviceMemory bvh_mem = VK_NULL_HANDLE;
  VkBuffer prim_buf = VK_NULL_HANDLE;
  VkDeviceMemory prim_mem = VK_NULL_HANDLE;

  // Compute pipeline
  VkDescriptorSetLayout comp_dsl = VK_NULL_HANDLE;
  VkDescriptorPool comp_pool = VK_NULL_HANDLE;
  VkDescriptorSet comp_set = VK_NULL_HANDLE;
  VkPipelineLayout comp_layout = VK_NULL_HANDLE;
  VkPipeline comp_pipeline = VK_NULL_HANDLE;

  // Display pipeline
  VkRenderPass disp_pass = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> disp_fbs;
  VkDescriptorSetLayout disp_dsl = VK_NULL_HANDLE;
  VkDescriptorPool disp_pool = VK_NULL_HANDLE;
  VkDescriptorSet disp_set = VK_NULL_HANDLE;
  VkPipelineLayout disp_layout = VK_NULL_HANDLE;
  VkPipeline disp_pipeline = VK_NULL_HANDLE;

  // Commands & sync
  VkCommandPool command_pool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> cmds;
  std::vector<VkSemaphore> img_available;
  std::vector<VkSemaphore> render_done;
  std::vector<VkFence> fences;

  bool validation_enabled = false;

  // Scene
  std::vector<GPUSphere> spheres;
  BVH bvh;
  uint32_t frame_index = 0;

  // Camera state
  glm::vec3 cam_pos = {0.0f, 3.0f, 10.0f};
  float cam_yaw = 0.0f;
  float cam_pitch = -0.10f;
  float cam_speed = 8.0f;
  float cam_fov = 40.0f;
  double last_time = 0.0;
  double last_mx = 0.0;
  double last_my = 0.0;
  bool first_mouse = true;

  // Init helpers
  void init_window();
  void init_vulkan();
  void main_loop();
  void cleanup();

  void create_instance();
  void setup_debug_messenger();
  void create_surface();
  void pick_physical_device();
  void create_logical_device();
  void create_swapchain();
  void create_render_images();
  void build_scene();
  void upload_scene();
  void create_compute_pipeline();
  void create_display_pipeline();
  void create_commands_and_sync();

  void draw_frame(uint32_t flight);
  bool process_input(float dt);
  void update_camera_ubo();

  // Vulkan utilities
  VkShaderModule load_shader(const std::string &path);
  uint32_t find_memory(uint32_t mask, VkMemoryPropertyFlags props);
  void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags props, VkBuffer &buf,
                     VkDeviceMemory &mem);
  void create_image2d(uint32_t w, uint32_t h, VkFormat fmt,
                      VkImageUsageFlags usage, VkImage &img,
                      VkDeviceMemory &mem);
  VkImageView create_image_view(VkImage img, VkFormat fmt);
  void transition_image(VkCommandBuffer cmd, VkImage img, VkImageLayout from,
                        VkImageLayout to);
  VkCommandBuffer begin_one_shot();
  void end_one_shot(VkCommandBuffer cmd);

  static std::vector<char> read_file(const std::string &path);
  static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
      VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
      const VkDebugUtilsMessengerCallbackDataEXT *, void *);
};
