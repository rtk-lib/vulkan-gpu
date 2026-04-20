BUILD := build

.PHONY: all clean

all: $(BUILD)/Makefile
	$(MAKE) -C $(BUILD) -j$(shell nproc 2>/dev/null || sysctl -n hw.logicalcpu)

$(BUILD)/Makefile:
	cmake -B $(BUILD)

clean:
	rm -rf $(BUILD)

format:
	@ find ./ -name "*.cpp" -type f -exec clang-format -i {} ";"
	@ find ./ -name "*.hpp" -type f -exec clang-format -i {} ";"
	@ $(LOG_TIME) "$(C_BLUE) CF $(C_GREEN) Code formated  $(C_RESET)"

check_format:
	@ find ./ -name "*.cpp" -type f -exec clang-format --dry-run --Werror {} ";" 2>&1 | wc -m | grep 0 > /dev/null
	@ find ./ -name "*.hpp" -type f -exec clang-format --dry-run --Werror {} ";" 2>&1 | wc -m | grep 0 > /dev/null
	@ $(LOG_TIME) "$(C_BLUE) CF $(C_GREEN) Code formated  $(C_RESET)"
