flags=-DPRETTY_PRINT -G -DGPU -DDEBUG -lm -arch sm_20
kmeans : kmeans.cu kmeans.h
	nvcc -o kmeans kmeans.cu $(flags)
clean :
	-rm -rf *.o *~ kmeans
