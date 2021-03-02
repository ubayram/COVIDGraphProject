# License

The content of this project is licensed under the MIT license. 2021 All rights reserved.


Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

Redistributions of source code must retain the above License notice, this list of conditions and the following disclaimers.

Redistributions in binary form must reproduce the above License notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR LICENSE HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

# Citation
These code are writen for a research project, published in OIR. If you use any of them, please cite:

Ulya Bayram, Runia Roy, Aqil Assalil, Lamia Ben Hiba, "The Unknown Knowns: A Graph-Based Approach for Temporal COVID-19 Literature Mining", Online Information Review (OIR), COVID-19 Special Issue, 2021.

# How to run the code:

In this study, we provided the data collection CORD19 as input. However, as our paper explains it in detail, these approaches and the scripts are applicable to other textual data collections as well. An important detail is that, while in the scripts we are providing some data files as inputs for the function calls, we had to delete all these files and the graphs we created due to copyright issues. Therefore, before running the scripts, feel free to take a look at the functions and change the input files according to your folder structures and file names.

1) The first code we run is called:

preprocess_cord19.py

This code calls the necessary functions to clean up the CORD19 collection (the version we used belonged to May 2020, such clean ups may no longer be necessary in the late versions of CORD19), and to re-collect and fix the metadata file provided with the collection. Next, it saves all the resulting, clean data files to proper folders.

2) The second code we run is called:

construct_graph.py

This code calls the necessary functions to use the previously cleaned up texts and to extract nodes and connections using intelligent methods that determines the technical terms present within the texts, finds the relations between these technical terms, and computes proper weights based on document and data collection level statistics between these terms. Then, it creates the undirected, weighted graph from these nodes and connections, and saves it to a folder called graphs/. Feel free to change the folder structures according to your needs.

3) The third code we run is called:

analyze_graph.py

This code calls the necessary functions to obtain the previously saved graph and performs simple graph analysis methods on it. Feel free to change which words you'd like analyze temporally etc.

4) The last code we run is called:

perform_link_prediction.py

This code performs the selected link prediction tehcniques over the sampled training/validation/test graphs. Writes the link prediction results to provided folders.
