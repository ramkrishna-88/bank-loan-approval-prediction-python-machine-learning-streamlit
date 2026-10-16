
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Bank Loan Approval", page_icon="🏦", layout="wide")
st.title("🏦 Bank Loan Approval Prediction")
st.caption("Week 4 Major Project • Random Forest / XGBoost-style classification workflow")

uploaded = st.sidebar.file_uploader("Upload Loan Prediction CSV", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else pd.read_csv("Loan_Prediction.csv")

required = [
    "Gender","Married","Dependents","Education","Self_Employed",
    "ApplicantIncome","CoapplicantIncome","LoanAmount",
    "Loan_Amount_Term","Credit_History","Property_Area","Loan_Status"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

df = df[required].copy().drop_duplicates()

st.subheader("1. Dataset Overview")
a,b,c,d = st.columns(4)
a.metric("Applications", len(df))
b.metric("Approved", int((df.Loan_Status=="Y").sum()))
c.metric("Rejected", int((df.Loan_Status=="N").sum()))
d.metric("Approval Rate", f"{(df.Loan_Status=='Y').mean()*100:.1f}%")

with st.expander("View dataset"):
    st.dataframe(df, use_container_width=True)

X = df.drop(columns="Loan_Status")
y = df["Loan_Status"].map({"N":0,"Y":1})

categorical = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
numeric = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
pipe = Pipeline([("preprocess", preprocess), ("model", model)])

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=.2,random_state=42,stratify=y
)
pipe.fit(X_train,y_train)
pred=pipe.predict(X_test)

st.subheader("2. Model Evaluation")
acc=accuracy_score(y_test,pred)
m1,m2=st.columns(2)
m1.metric("Accuracy",f"{acc:.3f}")
m2.metric("Test Applications",len(y_test))

cm=confusion_matrix(y_test,pred)
fig,ax=plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for (i,j),v in np.ndenumerate(cm):
    ax.text(j,i,str(v),ha="center",va="center")
st.pyplot(fig,clear_figure=True)

st.text("Classification Report")
st.text(classification_report(y_test,pred, target_names=["Rejected","Approved"], zero_division=0))

st.subheader("3. New Loan Application")
vals={}
cols=st.columns(3)
for i,c in enumerate(categorical):
    opts=list(df[c].dropna().unique())
    vals[c]=cols[i%3].selectbox(c,opts)

vals["ApplicantIncome"]=cols[0].number_input("ApplicantIncome",min_value=0,value=int(df.ApplicantIncome.median()))
vals["CoapplicantIncome"]=cols[1].number_input("CoapplicantIncome",min_value=0,value=int(df.CoapplicantIncome.median()))
vals["LoanAmount"]=cols[2].number_input("LoanAmount",min_value=1,value=int(df.LoanAmount.median()))
vals["Loan_Amount_Term"]=cols[0].number_input("Loan_Amount_Term",min_value=12,value=int(df.Loan_Amount_Term.median()))
vals["Credit_History"]=cols[1].selectbox("Credit_History",[0,1],index=1)

input_df=pd.DataFrame([vals])
if st.button("Check Loan Approval"):
    result=int(pipe.predict(input_df)[0])
    prob=float(pipe.predict_proba(input_df)[0,1])
    if result==1:
        st.success(f"Prediction: Loan Approved (model class Y). Approval probability: {prob:.3f}")
    else:
        st.warning(f"Prediction: Loan Not Approved (model class N). Approval probability: {prob:.3f}")

st.info("This is an educational machine-learning project. Real lending decisions require validated financial, legal and risk processes.")
