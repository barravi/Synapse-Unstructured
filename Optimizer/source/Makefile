#directories
OBJ = obj
SOU = src
INC = includes
BIN = bin

#commands
COMP = gcc -c -O3 -o
LINK = gcc -O3 -o

#files groups
EXECUTABLES  =  synopt
T_INCLUDES = ${INC}/const.h 
T_OBJECTS = ${OBJ}/alloc.o \
            ${OBJ}/gf.o
T_SOURCES = ${SOU}/alloc.c \
            ${SOU}/gf.c

#general rules
#${EXECUTABLES} : ${T_INCLUDES} ${T_SOURCES}

synopt : ${T_INCLUDES} ${T_OBJECTS} ${SOU}/main.c
	${LINK} synopt ${SOU}/main.c ${T_OBJECTS} -lm
	mv ${EXECUTABLES} ${BIN}

${OBJ}/alloc.o : ${SOU}/alloc.c ${T_INCLUDES}
	${COMP} alloc.o ${SOU}/alloc.c
	mv alloc.o ${OBJ}

${OBJ}/gf.o : ${SOU}/gf.c ${T_INCLUDES}
	${COMP} gf.o ${SOU}/gf.c
	mv gf.o ${OBJ}

clean :
	rm -f ${OBJ}/*.o
	rm -f ${BIN}/synopt
	
install :
	cp ${BIN}/${EXECUTABLES} ../
	