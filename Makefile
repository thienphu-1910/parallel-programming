CXX := g++

CPPFLAGS := -Iinclude -isystem third_party
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic

ifeq ($(OS),Windows_NT)
TARGET := stitch.exe
else
TARGET := stitch
endif

SOURCES := src/main.cpp \
           src/image_io.cpp \
           $(wildcard src/v0_sequential/*.cpp)

HEADERS := $(wildcard include/*.h) \
           third_party/stb_image.h \
           third_party/stb_image_write.h

.PHONY: all

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@