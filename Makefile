all:
	python -W ignore process.py

analyze:
	python -W ignore analyze.py

pass:
	cp plots/* paper/figures/
	cp csv/* paper/
