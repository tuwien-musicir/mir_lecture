![TUWien Logo](./img/TU_signet09_small.png)
![TUWien Logo](./img/ifs_logo_new.gif)
![TUWien Logo](./img/mir_logo3.jpg)

# Overview

Please help us to improve this script. If you discover errors, typos or other things we should correct or add to to the content, please create a ticket on the Github repository.

The interactive notebook is used in the lecture talk to demonstrate audio feature extraction, to visualize feature behavior as well as to demonstrate similarity retrieval and classification.

A nbconvert template is provided to convert the Notebook into a structured pdf. In the pdf-version all input code-cells are removed to improve readability. Only the outputs and charts are kept. You can find a converted pdf in the [converted_versions](https://github.com/tuwien-musicir/mir_lecture/tree/master/Textbook/converted_versions) directory.

Some parts are still work in progress. They are mentioned during the lecture and will be added to this textbook soon.


## Authors

### Alexander Schindler

Alexander Schindler is a researcher in the field of multimedia retrieval focusing on the audio-visual aspects of music information. As scientist at the Digital Safety and Security department of the AIT Austrian Institute of Technology he is working on audio indexing, classification and retrieval as well as image processing and digital preservation tasks. He is currently finishing his PhD thesis at department of Software Technology and Interactive Systems of the Vienna University of Technology. His research interests include information retrieval, specifically audio and video retrieval, deep learning and image processing. He has many years of experience in the field of Software Engineering in various companies as well as international projects.

Follow Alexander on:

* IFS Web-page: [http://www.ifs.tuwien.ac.at/~schindler/](http://www.ifs.tuwien.ac.at/~schindler/)
* Personal Homepage: [New](http://wwwnew.schindler.eu.com/), [Old](http://www.slychief.com/)
* Personal Twitter Channel: [https://twitter.com/slychief](https://twitter.com/slychief)
* Music-IR Twitter Channel (by Alexander Schindler): [https://twitter.com/music_ir](https://twitter.com/music_ir)
* LinkedIn: [https://at.linkedin.com/schindleralexander](https://at.linkedin.com/schindleralexander)


## Compiling the Textbook

	cd Textbook
	jupyter nbconvert --to pdf --template=mir_textbook_latex.tplx "Music Information Retrieval.ipynb"
