import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder    
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_df(df: pd.DataFrame, target: str, req):
    y = df[target]
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

    # Imputation
    num_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="mean"))
    ])
    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent"))
    ])

    # Encoding
    if req.encode == "label":
        cat_pipe.steps.append(("enc", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
    else:
        cat_pipe.steps.append(("enc", OneHotEncoder(handle_unknown="ignore")))
    
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ]
    )

    if req.scale != "none":
        scaler = StandardScaler() if req.scale == "standard" else MinMaxScaler()
        pre = Pipeline(steps=[("prep", pre), ("scale", scaler)])


    X_trans = pre.fit_transform(X)
    # Re-assemble
    cleaned = pd.DataFrame(X_trans)
    cleaned[target] = y.values
    return cleaned, pre