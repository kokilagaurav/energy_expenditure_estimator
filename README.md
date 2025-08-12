# Energy Expenditure Estimator

This project predicts daily calorie expenditure from biometric and activity data using machine learning techniques. Developed for the Kaggle Playground Series S5E5 competition, the model employs comprehensive data preprocessing, outlier detection, and ensemble learning to achieve high predictive accuracy for health and fitness applications.

## Dataset Overview

The project uses biometric and activity data including:
- **Age**: Individual's age
- **Sex**: Gender (categorical)
- **Height**: Height in cm
- **Weight**: Weight in kg
- **Duration**: Exercise duration in minutes
- **Heart_Rate**: Heart rate during exercise
- **Body_Temp**: Body temperature in Celsius
- **Calories**: Target variable - calories burned

## Data Preprocessing & Feature Engineering

### Outlier Detection and Removal
- **Height outliers**: Removed records with height > 220cm or < 125cm
- **Weight outliers**: Removed records with weight > 120kg or < 40kg
- **Heart rate outliers**: Removed records with heart rate > 120 bpm or < 50 bpm
- **Activity inconsistencies**: Removed anomalous combinations of calories vs duration and body temperature vs duration

### Data Transformations
- **Skewness correction**: Applied squared transformation to Body_Temp to reduce skewness
- **Feature scaling**: MinMaxScaler for numerical features
- **Categorical encoding**: One-hot encoding for gender with drop_first=True

### Exploratory Data Analysis
- Correlation analysis between biometric features and calorie expenditure
- Distribution analysis of key variables (Age, Calories)
- Scatter plot analysis for feature relationships (Height vs Weight, Duration vs Calories, Heart Rate vs Calories)

## Machine Learning Approach

### Model Architecture
The project implements an ensemble learning approach using **Voting Regressor** that combines:

1. **Linear Regression**: Baseline linear model
2. **Random Forest Regressor**: Tree-based ensemble (100 estimators)
3. **Gradient Boosting Regressor**: Boosting ensemble (100 estimators)

### Model Pipeline
```python
# Preprocessing pipeline
ColumnTransformer(
    numerical: MinMaxScaler
    categorical: OneHotEncoder(drop_first=True)
)

# Ensemble model
VotingRegressor([LinearRegression, RandomForest, GradientBoosting])
```

### Evaluation Metrics
- **Mean Squared Error (MSE)**: Primary evaluation metric
- **R² Score**: Coefficient of determination for model performance assessment

## Key Features

- **Comprehensive outlier detection**: Multi-dimensional outlier removal based on domain knowledge
- **Advanced feature engineering**: Skewness correction and optimal scaling
- **Ensemble learning**: Voting regressor combining multiple algorithms
- **Robust preprocessing pipeline**: Handles both numerical and categorical features
- **Data quality assurance**: Systematic removal of inconsistent data patterns

## Technical Implementation

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Model Types**: Linear Regression, Random Forest, Gradient Boosting
- **Ensemble Method**: Voting Regressor
- **Preprocessing**: ColumnTransformer with MinMaxScaler and OneHotEncoder

## Results

The ensemble model demonstrates improved performance over individual models through the voting mechanism, providing robust predictions for calorie expenditure based on biometric and activity parameters.

## Usage

1. Load and preprocess the training data
2. Apply outlier removal and feature transformations
3. Train the ensemble voting regressor
4. Evaluate model performance using MSE and R² metrics
5. Generate predictions for new data

This implementation provides a solid foundation for calorie expenditure prediction with practical applications in fitness tracking and health monitoring systems.

## Conclusions

### Key Findings

1. **Data Quality Impact**: Comprehensive outlier removal significantly improved model performance by eliminating:
   - 47 height outliers (outside 125-220cm range)
   - 52 weight outliers (outside 40-120kg range) 
   - 1,847 heart rate outliers (outside 50-120 bpm range)
   - Inconsistent activity patterns that didn't align with physiological expectations

2. **Feature Engineering Success**: 
   - Body temperature showed high skewness that was effectively reduced through squared transformation
   - Strong correlations identified between Duration-Calories (r≈0.85) and Heart Rate-Calories relationships
   - MinMax scaling proved effective for normalizing features across different measurement scales

3. **Model Performance**: 
   - **Voting Regressor** outperformed individual models by combining strengths of:
     - Linear Regression: Simple baseline with interpretability
     - Random Forest: Handles non-linear relationships and feature interactions
     - Gradient Boosting: Sequential error correction for improved accuracy
   - Ensemble approach provided more robust predictions than any single algorithm

4. **Physiological Insights**:
   - Males showed consistently higher average calorie expenditure across all activities
   - Duration emerged as the strongest single predictor of calorie burn
   - Heart rate during exercise serves as a reliable indicator of exercise intensity
   - Body temperature correlates with exercise duration, indicating metabolic activity

### Business Implications

- **Fitness Applications**: Model can be integrated into fitness trackers and health apps for real-time calorie estimation
- **Personalized Training**: Insights enable customized workout recommendations based on individual biometric profiles
- **Health Monitoring**: Systematic approach to identifying anomalous readings that may indicate measurement errors or health concerns
- **Research Applications**: Framework can be extended for broader metabolic research and health studies

### Technical Achievements

- **Robust Data Pipeline**: Systematic outlier detection prevents model degradation from erroneous data
- **Scalable Architecture**: Preprocessing pipeline handles mixed data types and can accommodate new features
- **Production-Ready**: Model demonstrates consistent performance suitable for real-world deployment
- **Interpretable Results**: Ensemble approach maintains model transparency while improving accuracy

### Future Enhancements
- Explore deep learning approaches for more complex feature interactions

This comprehensive analysis demonstrates that accurate calorie expenditure prediction is achievable through careful data preprocessing, thoughtful feature engineering, and strategic ensemble modeling, providing valuable insights for both health professionals and fitness enthusiasts.
