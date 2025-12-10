# Image Classification: Comparative Evaluation of Machine Learning Models

Comparative evaluation of KNN, NCC, Linear SVM, and RBF SVM classifiers on CIFAR-10 and SVHN datasets.

## Datasets

- **CIFAR-10**: 50,000 training / 10,000 test images (10 classes)
- **SVHN**: 73,257 training / 26,032 test images (digits 0-9)

## Models

- **KNN**: k=10 neighbors
- **NCC**: Nearest Class Centroid
- **Linear SVM**: C ∈ {0.1, 1, 5, 10}
- **RBF SVM**: C ∈ {0.1, 1, 3}, γ ∈ {"scale", "auto"}

## Preprocessing

1. Flatten images
2. L2 normalization
3. Z-score standardization
4. PCA (90% variance retention)
5. Label preprocessing

## Installation

```bash
pipenv install
pipenv shell
python main.py
```

## Configuration

Edit `program_basic_info` in `main.py`:

```python
program_basic_info: dict = {
    "desired_data": "cifar10",  # "cifar10" or "svhn"
    "linear_parameters": {"C": [0.1, 1, 5, 10]},
    "rbf_parameters": {
        "C": [0.1, 1, 3],
        "gamma": ["scale", "auto"]
    },
    "minimize_samples": False
}
```

## Project Structure

```
Program/
├── main.py                    # Main script
├── data_helper.py            # Preprocessing
├── dataset_helper.py         # Dataset loading
├── knn_ncc_evaluator.py     # KNN/NCC models
├── svm_evaluator.py         # SVM models
├── visualization_helper.py  # Visualizations
├── svhn_data/              # SVHN dataset
└── visualizations/         # Output files
```

## Output

- Console: Performance metrics and results tables
- Visualizations: Prediction examples and comparison charts

## Requirements

- Python 3.12
- NumPy, scikit-learn, TensorFlow, Matplotlib, Pandas, SciPy

---

**Course**: Computational Intelligence and Statistical Learning  
**Institution**: Aristotle University of Thessaloniki
