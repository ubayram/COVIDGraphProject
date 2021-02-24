To uncompress the files in the data:

cat compressed.gz* | zcat > /path/to/decrompressed/file


preprocess_cord19.py

As guided by the code, provide the data collection (CORD19 used in this study) as input. Scipts preprocesses the texts, and collects appropriate metadata.
