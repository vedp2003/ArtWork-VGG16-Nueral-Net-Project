# Predicting Artwork Creation Years - Machine Learning & Statistical Analysis

This project explores the prediction of artwork creation years using embeddings extracted from the VGG-16 convolutional neural network. It combined statistical analysis, dimensionality reduction, and machine learning to build robust predictive models.

---

## Dataset

The dataset contains embeddings extracted from the VGG-16 neural network and corresponding artwork creation years. These embeddings represent high-dimensional features derived from images of artworks, requiring dimensionality reduction and preprocessing.

---

## Key Steps Implemented in this Project

### Data Preprocessing and Cleaning
1. **Embedding Normalization**: Mean-centered and whitened the embeddings to ensure standardization and reduce correlations.
2. **Dimensionality Reduction**: Applied Principal Component Analysis (PCA) to reduce the high-dimensional embeddings to 1-2 dimensions while retaining significant variance.
3. **Outlier Handling**: Managed extreme deviations in the dataset to improve model reliability.

### Feature Engineering
1. **PCA Components**: 
   - One-dimensional PCA: Used for initial visualization of linear relationships.
   - Two-dimensional PCA: Enhanced modeling through additional variance capture.
2. **Data Normalization**: Scaled PCA components and year values using MinMaxScaler for consistent range compatibility with spline fitting.

### Exploratory Data Analysis (EDA)
1. Visualized trends between PCA components and artwork creation years.
2. Assessed linear and non-linear relationships through scatter plots and 3D visualizations.

### Statistical and Machine Learning Models

#### Polynomial Regression
1. Explored varying polynomial degrees (1-8) for optimal bias-variance tradeoff.
2. Achieved the lowest training Mean Squared Error (MSE) with a degree of 8.

#### Smoothing Splines
1. Applied univariate smoothing splines to each PCA component.
2. Tuned the smoothness parameter to balance model complexity and data fit.
3. Averaged predictions from splines for final outputs.

### Statistical Enhancements
1. **Bootstrapping**:
   - Resampled training data 100 times to estimate variability and calculate confidence intervals for MSE.
   - Result: Mean MSE of ~16,200 with a 95% confidence interval of [15,500, 17,100].
2. **False Discovery Rate (FDR) Control**:
   - Reduced features using the Benjamini-Hochberg procedure to retain only statistically significant predictors.

### Model Evaluation
1. Evaluated performance using metrics such as MSE and variance explained (R²).
2. Analyzed most and least accurate predictions to understand model limitations.

---

## Flow of the Project

1. **Setup and Environment**:
   - Clone this repository and ensure Python 3.x is installed.

2. **Data Import**:
   - Load the dataset containing VGG-16 embeddings and artwork years.

3. **Data Preprocessing**:
   - Normalize embeddings and reduce dimensionality using PCA.

4. **Visualization**:
   - Create visualizations to understand relationships between PCA components and target variables.

5. **Modeling**:
   - Train polynomial regression and smoothing splines models.
   - Validate using bootstrap resampling and FDR control.

6. **Prediction**:
   - Use the trained models to predict artwork creation years from embeddings.

---

## Requirements

The following Python libraries are used in the project:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `scipy`

---

## Steps to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/vedp2003/ArtWork-VGG16-Nueral-Net-Project.git
   ```

2. Navigate to the project folder:
   ```bash
   cd ArtWork-VGG16-Nueral-Net-Project
   ```

3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook ArtworkYearPrediction.ipynb
   ```

5. Follow the notebook cells to:
   - Preprocess the data
   - Train the models
   - Visualize results
   - Evaluate model performance

---

## Key Takeaways

1. Dimensionality reduction through PCA simplified the dataset while retaining key trends.
2. Polynomial regression and smoothing splines effectively captured relationships in the data.
3. Statistical methods, including bootstrapping and FDR control, improved model reliability and interpretability.
4. This project highlights the intersection of art history, machine learning, and statistical analysis, showcasing the potential of predictive analytics in cultural domains.
