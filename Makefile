# Makefile to build library 'openair~' for Pure Data.
# Needs Makefile.pdlibbuilder as helper makefile for platform-dependent build
# settings and rules.
#
# use : make pdincludepath=/path/to/pure-data/src/
#
# The following command will build the external and install the distributable 
# files into a subdirectory called build/moonlib :
#
# make install pdincludepath=../pure-data/src/ objectsdir=./build

lib.name = openair

objects = openair~

# link statically
openair~.class.ldlibs = -l:libfftw3f.a

class.sources = $(addsuffix .c,$(objects))

# all extra files to be included in binary distribution of the library
datafiles = $(addsuffix -help.pd,$(objects)) LICENSE.txt README.txt
datadirs = examples

cflags = -Wno-unused -Wno-unused-parameter 

# include Makefile.pdlibbuilder from submodule directory 'pd-lib-builder'
PDLIBBUILDER_DIR=pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder
