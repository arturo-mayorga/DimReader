# DimReader
Axis lines that explain non-linear projections

# Getting Started
TLDR;
```
cd src
python DimReaderPlot.py dim-reader-sample_output.csv IRIS-labels.csv 0 3
```

Navigate to the src directory and generate a DimReader output file.  Note, this takes a while.
```
cd src
python RimReader.py IRIS.csv all tsne
```

This will generate a file that looks like `2018-10-25_08:12:49_output.csv`.  You can then plot this file or the provided `dim-reader-sample_output.csv` if you want to skip the previous step.

```
python DimReaderPlot.py dim-reader-sample_output.csv IRIS-labels.csv 0 3
```

# Acknowledgements
The code in this repo is based on the additional materials attached to the ieee vis 2018 proceedings for the paper DimReader: Axis lines that explain non-linear projections by Rebecca Faust, David Glickenstein, and Carlos Scheidegger.

# Paper Abstract
Non-linear dimensionality reduction (NDR) methods such as LLE and t-SNE are popular with visualization researchers and experienced data analysts, but present serious problems of interpretation. In this paper, we present DimReader, a technique that recovers readable axes from such techniques. DimReader is based on analyzing inﬁnitesimal perturbations of the dataset with respect to variables of interest. The perturbations deﬁne exactly how we want to change each point in the original dataset and we measure the effect that these changes have on the projection. The recovered axes are in direct analogy with the axis lines (grid lines) of traditional scatterplots. We also present methods for discovering perturbations on the input data that change the projection the most. The calculation of the perturbations is efﬁcient and easily integrated into programs written in modern programming languages. We present results of DimReader on a variety of NDR methods and datasets both synthetic and real-life, and show how it can be used to compare different NDR methods. Finally, we discuss limitations of our proposal and situations where further research is needed.
