![TUWien Logo](./Textbook/img/TU_signet09_small.png)
![TUWien Logo](./Textbook/img/ifs_logo_new.gif)
![TUWien Logo](./Textbook/img/mir_logo3.jpg)

# Overview

This repository contains relevant resources for teaching Music Information Retrieval. The main contribution is the executable textbook. This script is implemented using a Jupyther Notebook and is intended to be used for the lecture talk as well as to be converted into formatted textbook.

## TU-Wien Music Information Retrieval Team

Our team has a strong background in Information Retrieval in general, but particularly also in Music Information Retrieval, since 1999.

Our research focuses on various methods of indexing and structuring audio collections, as well as providing intuitive user interfaces to facilitate exploration of musical libraries. Machine learning techniques are used to e xtract semantic information, group audio by similarity, or classify it into various categories. We developed advanced visualization techniques to provide intuitive interfaces to audio collections on standard PC as well as mobile devices. Our solutions automatically organize music according to its sound characteristics such that we find similar pieces of music grouped together, enabling direct access to and intuitive instant playback according to one's current mood.

* Group Web-page: [http://www.ifs.tuwien.ac.at/mir/](http://www.ifs.tuwien.ac.at/mir/)
* Audio Feature Extraction resources [http://www.ifs.tuwien.ac.at/mir/audiofeatureextraction.html](http://www.ifs.tuwien.ac.at/mir/audiofeatureextraction.html)
* RP-Extract on Github [https://github.com/tuwien-musicir/rp_extract](https://github.com/tuwien-musicir/rp_extract)


* Music Information Retrieval Wiki [http://wiki.schindler.eu.com/doku.php?id=start](http://wiki.schindler.eu.com/doku.php?id=start)
* Music-IR Twitter Channel [https://twitter.com/music_ir](https://twitter.com/music_ir)


## Compiling the Textbook

	cd Textbook
	jupyter nbconvert --to pdf --template=mir_textbook_latex.tplx "Music Information Retrieval.ipynb"
