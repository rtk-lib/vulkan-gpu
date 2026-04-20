BUILD := build

.PHONY: all clean

all: $(BUILD)/Makefile
	@ $(MAKE) -C $(BUILD) -j$(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu)

$(BUILD)/Makefile:
	@ cmake -B $(BUILD)

include utils.mk

clean:
	@ rm -rf build
	@ $(LOG_TIME) "$(C_YELLOW) RM $(C_YELLOW) build/  $(C_RESET)"

format:
	@ find ./ -name "*.cpp" -type f -exec clang-format -i {} ";"
	@ find ./ -name "*.hpp" -type f -exec clang-format -i {} ";"
	@ clang-format -i shaders/*
	@ $(LOG_TIME) "$(C_BLUE) CF $(C_GREEN) Code formated  $(C_RESET)"

check_format:
	@ find ./ -name "*.cpp" -type f -exec clang-format --dry-run --Werror {} ";" 2>&1 | wc -m | grep 0 > /dev/null
	@ find ./ -name "*.hpp" -type f -exec clang-format --dry-run --Werror {} ";" 2>&1 | wc -m | grep 0 > /dev/null
	@ $(LOG_TIME) "$(C_BLUE) CF $(C_GREEN) Code formated  $(C_RESET)"
