CXX?=g++
CXXFLAGS=-O1 -Iinclude -I/usr/include/opencv4/

# Libraries
LDFLAGS += -llapack -lblas -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio

EXEC=hw2

INCLUDEDIR=include
SRCDIR=src
OBJDIR=objs

#OPENCV_INCLUDE_PATH = `pkg-config --cflags opencv`
#OPENCV_LIBS = `pkg-config --libs opencv`

LIBS=-lc $(OPENCV_LIBS)

SRCS=$(wildcard $(SRCDIR)/*.cxx)
OBJS=$(patsubst $(SRCDIR)/*.cxx, $(OBJDIR)/*.o, $(SRCS))
INCLUDES=-I$(INCLUDEDIR) $(OPENCV_INCLUDE_PATH)

all: $(SRCS) $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(EXEC) $(LIBS) $(LDFLAGS)

%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $< -o $@
install:
	insmod CPUcntr.ko
	dmesg|tail -1
clean:
	rm -f $(EXEC)
	rm -f $(OBJDIR)/*.
	rmmod CPUcntr.ko
