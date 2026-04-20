#include "vulkan_app.h"
#include <iostream>
#include <stdexcept>

int main() {
  try {
    VulkanApp app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
