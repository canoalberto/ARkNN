# Aging and rejuvenating strategies for fading windows in multi-label classification on data streams

Combining the challenges of streaming data and multi-label learning, the task of mining a drifting, multi-label data stream requires methods that can accurately predict labelsets, adapt to various types of concept drift and run fast enough to process each data point before the next arrives. To achieve greater accuracy, many multi-label algorithms use computationally expensive techniques, such as multiple adaptive windows, with little concern for runtime and memory complexity. We present Aging and Rejuvenating kNN (ARkNN) which uses simple resources and efficient strategies to weight instances based on age, predictive performance, and similarity to the incoming data. We break down ARkNN into its component strategies to show the impact of each and experimentally compare ARkNN to seven state-of-the-art methods for learning from multi-label data streams. We demonstrate that it is possible to achieve competitive performance in multi-label classification on streams without sacrificing runtime and memory use, and without using complex and computationally expensive dual memory strategies.

## Using ARkNN

Download the pre-compiled `ARkNN-1.0-jar-with-dependencies.jar` or import the project source code into [MOA](https://github.com/Waikato/moa). Download the multi-label [datasets](https://drive.google.com/file/d/1eB3T70aagGSZjSmg4t4s0yIxEjWxLdmz/view?usp=sharing). When changing the dataset file please make sure to also update the parameter `-c` to the number of labels in the dataset. The `c` labels must be the first `c` columns of the dataset.

```
java -javaagent:sizeofag-1.0.4.jar -cp ARkNN-1.0-jar-with-dependencies.jar moa.DoTask EvaluatePrequentialMultiLabel -e "(PrequentialMultiLabelPerformanceEvaluator)" -s "(MultiTargetArffFileStream -c 6 -f datasets/Scene.arff)" -l "(moa.classifiers.multilabel.ARkNN)" -f 100 -d results.csv
```

The package `src/main/java/experiments` provides the scripts to generate the experiments and collect results for all datasets and algorithms.

## Citation
```
@inproceedings{roseberry2023SAC,
  title={Aging and rejuvenating strategies for fading windows in multi-label classification on data streams},
  author={Roseberry, Martha and D{\v{z}}eroski, Sa{\v{s}}o and Bifet, Albert and Cano, Alberto},
  booktitle={38th ACM/SIGAPP Symposium On Applied Computing},
  year={2023}
}
```