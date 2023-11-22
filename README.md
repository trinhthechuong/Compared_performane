# Statistical Test for Machine Learning Model Performance Comparison

## Overview

This repository offers a Python `Statistical_test` package for comparing the performance of machine learning models through statistical tests. Two approaches are implemented: Bayesian estimation and the Wilcoxon signed-rank test. 

## Code example
```python
# Import the Statistical package and statistical_test function
from Statistical_test import statistical_test

# Input the list of models cross-validation scores, names of your implemented models, X_train, and y_train with your actual data and training sets
model_compare = statistical_test(results=list_results, model=name, X_train=X_train, y_train=y_train)

# Bayesian estimation
bayesian_df = model_compare.bayesian_comparison()

# Wilcoxon signed-rank test
df_pvalue = model_compare.wilcoxon_comparison()  # Calculating p-value

# Boxplot visualization
model_compare.boxplot_comparision(show_pvalue=True)

# Heatmap visualization
model_compare.posthoc_comparison(title="Comparison heatmap Wilcoxon", save=False)
```

## Example notebook
Check out the detailed example in the [example notebook](Statistical_test_ML_performance.ipynb) for a step-by-step guide and comprehensive examples.

## Contributing
Feel free to contribute by opening issues or submitting pull requests. Contributions are welcome!

## Acknowledgments
Special thanks to [Tieu-Long PHAN](https://tieulongphan.github.io/cv/) for inspiration and guidance.

## References
1. Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. The Journal of Machine Learning Research, 18(1).
2. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. The Journal of Machine learning research, 7, 1-30.