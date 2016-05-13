![TUWien Logo](./Textbook/img/TU_signet09.png | width=100)

# Overview

This repository contains relevant resources for teaching Music Information Retrieval. The main contribution is the executable textbook. This script is implemented using a Jupyther Notebook and is intended to be used for the lecture talk as well as to be converted into formatted textbook.


# Compiling the Textbook

	cd Textbook
	jupyter nbconvert --to pdf --template=mir_textbook_latex.tplx "Music Information Retrieval.ipynb"
