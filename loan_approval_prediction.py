
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Loan_Prediction.csv").drop_duplicates()

features = [
    "Gender","Married","Dependents","Education","Self_Employed",
    "ApplicantIncome","CoapplicantIncome","LoanAmount",
    "Loan_Amount_Term","Credit_History","Property_Area"
]
X = df[features]
y = df["Loan_Status"].map({"N":0,"Y":1})

categorical = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
numeric = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))
])

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=.2,random_state=42,stratify=y
)
model.fit(X_train,y_train)
pred=model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))
print("Confusion Matrix:\n", confusion_matrix(y_test,pred))
print(classification_report(y_test,pred,target_names=["Rejected","Approved"],zero_division=0))
