#include "Logger/Logger.hpp"
#include "vulkan_app.h"

int main() {
  try {
    VulkanApp app;
    app.run();
  } catch (const std::exception &e) {
    return LOG_FATAL(e.what()), EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
