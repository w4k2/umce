all:
	python -W ignore analyze.py
	python -W ignore process.py
	cp plots/* paper/figures/
	cp csv/* paper/
