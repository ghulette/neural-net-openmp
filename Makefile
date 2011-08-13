CC=g++

compile:
	$(CC) *.cpp -o nns -O3

compile-openmp:
	icc *.cpp -o nnm -openmp -O3

clean:
	rm -f nnm nns
