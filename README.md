# diabetes-prediction-ml
chose logistic regression for medical interpretability.

#  Diabetes Prediction with Machine Learning

##  Project Overview  
**Predicts diabetes risk** with **75% accuracy** using patient health metrics. Designed for healthcare applications where model interpretability is crucial.

##  Key Features  
- **Data Preprocessing**: Handled missing values, applied Standard Scaling  
- **Model**: Logistic Regression (chosen for medical interpretability)  
- **Evaluation**: 75% accuracy, F1-score of 0.66 for diabetic cases  
- **Visualizations**: Feature importance and confusion matrix analysis  

## Results  
![Confusion Matrix](confusion_matrix.png)  
*Model correctly identified 79/99 healthy patients and 37/55 diabetic cases*  

##  Technical Stack  
- Python 3.8  
- Pandas, Scikit-learn, Seaborn  
- Jupyter Notebook  

##  How to Run  
```bash
git clone https://github.com/AhmadSohail2006/diabetes-prediction-ml.git
pip install -r requirements.txt
jupyter notebook LogisticRegression.ipynb
