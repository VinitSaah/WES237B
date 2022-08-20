CXX?=g++
CXXFLAGS=-Ofast -std=c++11 -mfpu=neon -O1 -ftree-vectorize -pg

EXEC=lab3_fir

INCLUDEDIR=include
SRCDIR=src
OBJDIR=objs

#OPENCV_INCLUDE_PATH = `pkg-config --cflags opencv`
#OPENCV_LIBS = `pkg-config --libs opencv`

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
	rm gmon.out
