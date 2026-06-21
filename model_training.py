import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression,Ridge, Lasso, ElasticNet,LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report, auc, roc_auc_score




df = pd.read_csv('diabetes_prediction_dataset.csv')
df.sample(5)
df.shape
df.columns
df.isnull().mean()
df.duplicated().sum()
df = df.drop_duplicates()
df.duplicated().sum()
df.describe()
df.info()



# EDA
sns.countplot(data=df, x='diabetes')
plt.show()
df.hist(figsize=(12,10), bins=20)
plt.suptitle("Feature Distributions")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only = True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()



df['diabetes'].value_counts(normalize = True)
X = df.drop("diabetes",axis = 1)
y = df["diabetes"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 42, stratify = y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
num = X.select_dtypes(exclude = ['object']).columns
cat = X.select_dtypes(include = ['object']).columns
process = ColumnTransformer(   
    transformers = [("num",StandardScaler(),num),
                     ("cat",OneHotEncoder(handle_unknown = "ignore"),cat)
                   ]
)



## Random Forest Classifier
model1 = RandomForestClassifier(n_estimators = 200,max_depth = 8,random_state = 42)
clf1 = Pipeline(steps=[("preprocessor",process),("model1",model1)])
clf1.fit(X_train,y_train)
y_pred1 = clf1.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Classification Report:\n", classification_report(y_test, y_pred1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
y_prob_rf = clf1.predict_proba(X_test)[:,1]
print("RF AUC:", roc_auc_score(y_test, y_prob_rf))




## Logistic Regression
model2 = LogisticRegression(max_iter = 1000)
clf2 = Pipeline(steps=[("preprocessor",process),("model2",model2)])
clf2.fit(X_train,y_train)
y_pred2 = clf2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Classification Report:\n", classification_report(y_test, y_pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
y_prob_lr = clf2.predict_proba(X_test)[:,1]
print("LR AUC:", roc_auc_score(y_test, y_prob_lr))





# Choosing RandomForetClissifier as it has higher accuracy.
# Predicting
def predict_diabetes(gender, age, hypertension, heart_disease,
                     smoking_history, bmi, HbA1c_level, blood_glucose_level):

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    prediction = clf1.predict(input_data)[0]

    return "Diabetes" if prediction == 1 else "No diabetes"


print(predict_diabetes("Female", 80, 0, 1, "never", 25.19, 6.6, 140))