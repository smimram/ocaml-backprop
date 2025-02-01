all:
	@dune build

test:
	@dune exec -- ./amp.exe -s source.wav -t target.wav

clone:
	git clone git@github.com:Alec-Wright/Automated-GuitarAmpModelling.git
	ln -s Automated-GuitarAmpModelling/Data/train/ht1-input.wav source.wav
	ln -s Automated-GuitarAmpModelling/Data/train/ht1-target.wav target.wav
