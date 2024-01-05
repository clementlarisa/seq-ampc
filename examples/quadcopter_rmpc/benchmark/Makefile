CC = g++
# CXXFLAGS  = -std=c++20  -g -Wall -O3 -I -lstdc++ -msse2 -ffast-math -fassociative-math -fopt-info-vec -fgraphite-identity -ftree-loop-distribution -floop-nest-optimize -march=znver3
CXXFLAGS  = -std=c++20  -g -Wall -O3 -I -lstdc++ -mavx2 -msse2 -mfma -march=znver3
CXXFLAGS += -I/usr/local/include/eigen3
CXXFLAGS += -I../c_generated_code

INCLUDE_PATH = /home/hose/projects/dsme/forks/acados/include
LIB_PATH = /home/hose/projects/dsme/forks/acados/lib

# preprocessor flags for make's implicit rules
CPPFLAGS+= -I$(INCLUDE_PATH)
CPPFLAGS+= -I$(INCLUDE_PATH)/acados
CPPFLAGS+= -I$(INCLUDE_PATH)/blasfeo/include
CPPFLAGS+= -I$(INCLUDE_PATH)/hpipm/include


# define the c-compiler flags for make's implicit rules
CFLAGS = -fPIC -std=c99   -O3#-fno-diagnostics-show-line-numbers -g
# # Debugging
# CFLAGS += -g3

# linker flags
LDFLAGS+= -L$(LIB_PATH)
LDFLAGS+= -L../c_generated_code

# link to libraries
LDLIBS+= -lacados
LDLIBS+= -lhpipm
LDLIBS+= -lblasfeo
LDLIBS+= -lm
LDLIBS+= -l:libacados_sim_solver_quadcopter.so
LDLIBS+= 

DEPS = neural_network.hpp 
OBJ = main.o 

%.o: %.c $(DEPS)
	$(CC) $(CXXFLAGS) -c -o $@ $< 

main: $(OBJ)
	$(CC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)