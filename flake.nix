{
  description = "GPU Raytracer — Vulkan compute + BVH";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs   = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "gpu-raytracer";

      buildInputs = with pkgs; [
        cmake
        ninja
        pkg-config

        # Vulkan
        vulkan-headers
        vulkan-loader
        vulkan-validation-layers
        vulkan-tools
        shaderc               # glslc

        # Window / math
        glfw
        glm

        # Debug
        gdb
      ];

      shellHook = ''
        export VULKAN_SDK="${pkgs.vulkan-loader}"
        export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
        echo "GPU raytracer dev shell ready"
        echo "  cmake -B build -G Ninja && cmake --build build -j\$(nproc)"
        echo "  ./build/raytracer"
      '';
    };
  };
}
