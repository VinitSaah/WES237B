CXX?=g++
CXXFLAGS=-O3 -std=c++11 -g
NVCC?=nvcc
NVCCFLAGS=-O3 -Wno-deprecated-gpu-targets

EXEC=lab4

INCLUDEDIR=include
SRCDIR=src
OBJDIR=objs

OPENCV_VERSION := $(shell pkg-config --modversion opencv4)

ifeq "$(OPENCV_VERSION)" "4.1.1"
	OPENCV_INCLUDE_PATH = `pkg-config --cflags opencv4`
	OPENCV_LIBS = `pkg-config --libs opencv4`
	CXXFLAGS += -DOPENCV4
else
	OPENCV_INCLUDE_PATH = `pkg-config --cflags opencv`
	OPENCV_LIBS = `pkg-config --libs opencv`
endif

CUDA_LIBS=-L/usr/local/cuda/lib64 -lcudart
CUDA_INCLUDE=-I/usr/local/cuda/include

LIBS=-lc $(OPENCV_LIBS) $(CUDA_LIBS)
INCLUDES=-I$(INCLUDEDIR) $(OPENCV_INCLUDE_PATH) $(CUDA_INCLUDE)

SRCS=$(wildcard $(SRCDIR)/*.cpp)
OBJS=$(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

SRCS_CUDA=$(wildcard $(SRCDIR)/*.cu)
OBJS_CUDA=$(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(SRCS_CUDA))

all: $(EXEC)
	@echo $(OPENCV_VERSION)

$(EXEC): $(OBJS) $(OBJS_CUDA)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) $(OBJS_CUDA) -o $(EXEC) $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(EXEC)
	rm -f $(OBJDIR)/*.o

