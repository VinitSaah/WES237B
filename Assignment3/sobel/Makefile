CXX?=g++
CXXFLAGS=-O1 -ftree-vectorize -std=c++11 -I/usr/include/opencv4/ -g

EXEC=hw3

INCLUDEDIR=include
SRCDIR=src
OBJDIR=objs

#OPENCV_INCLUDE_PATH = /usr/include/opencv4/
#OPENCV_LIBS = `pkg-config --libs opencv`

LDFLAGS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio

LIBS=-lc $(OPENCV_LIBS)
INCLUDES=-I$(INCLUDEDIR) $(OPENCV_INCLUDE_PATH)

SRCS=$(wildcard $(SRCDIR)/*.cpp)
OBJS=$(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(EXEC) $(LIBS) $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(EXEC)
	rm -f $(OBJDIR)/*.o
	rm -f ./image_outputs/*.tif
