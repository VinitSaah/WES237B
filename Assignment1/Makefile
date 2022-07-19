CXX?=g++
CXXFLAGS=-O1 -Wall

EXEC=CODEC

INCLUDEDIR=include
SRCDIR=src
OBJDIR=objs

OPENCV_INCLUDE_PATH = `pkg-config --cflags opencv4`
OPENCV_LIBS = `pkg-config --libs opencv4`

LIBS=-lc $(OPENCV_LIBS)

SRCS=$(wildcard $(SRCDIR)/*.cpp)
OBJS=$(patsubst $(SRCDIR)/*.cpp, $(OBJDIR)/*.o, $(SRCS))
INCLUDES=-I$(INCLUDEDIR) $(OPENCV_INCLUDE_PATH)

all: $(SRCS) $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) -o $(EXEC) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(EXEC)
	rm -f $(OBJDIR)/*.o
