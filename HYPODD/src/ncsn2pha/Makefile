CMD	= ncsn2pha
CC	= gcc
FC	= gfortran 
#FC	= f77
SRCS	= $(CMD).f
CSRCS	= 
OBJS	= $(SRCS:%.f=%.o) $(CSRCS:%.c=%.o)
CFLAGS	= -O
#FFLAGS  = -O -fno-silent -Wall -implicit
FFLAGS	= -O

all: $(CMD)

$(CMD): $(OBJS)
	$(FC) $(LDFLAGS) $(OBJS) $(LIBS) -o $@

%.o: %.f
	$(FC) $(FFLAGS) -c $(@F:.o=.f) -o $@

# Extensive lint-like diagnostic listing (f77 only)
$(CMD).lst: $(SRCS)
	f77 -e -Xlist -c $(SRCS)

clean:
	-rm -f $(CMD) *.o core a.out *.lint junk

atoangle_.o : compat.h
atoangle_.o : f77types.h
ph2dt.o : ph2dt.inc
