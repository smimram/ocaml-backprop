all: data
	@dune build

test: data
	@dune exec ./mnist.exe

data:
	git clone git@github.com:fgnt/mnist.git $@
	cd $@ && gunzip *.gz
