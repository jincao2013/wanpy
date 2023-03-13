obj =  greenTB.o

f90  = ifort -fpp -DMPI

flag   =  -O2 -nogen-interface

libs = -L/opt/intel2016/mkl/lib/intel64/ \
		-lmkl_intel_lp64 -lmkl_sequential \
		-lmkl_core -liomp5

main :  $(obj)
	    $(f90) $(flag) -o green.x $(obj) $(libs)
	    $(clean)

.SUFFIXES: .o .f90
.f90.o :
$(f90) -c $(flag) $(includes) $*.f90

clean :
	rm -f *.o *.mod 

