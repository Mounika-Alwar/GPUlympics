from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import pandas as pd
from utils import map_model_size

df = pd.read_csv("../synthetic_gpu_training.csv")

# Define preprocessing pipeline
ordinal_transformer = FunctionTransformer(map_model_size,validate=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('gpu', OneHotEncoder(), ['gpu_type'])
    ],
    remainder='passthrough'
)
full_preprocessing = Pipeline([
    ('model_mapping', ordinal_transformer),
    ('one_hot', preprocessor)
])

X = df[['gpu_type','model_size','batch_size','learning_rate','seq_length','run_id']]
full_preprocessing.fit(X)

model = joblib.load(os.path.join("best_model.pkl")) 

bundle = {
    'model': model,
    'preprocessor': full_preprocessing
}

joblib.dump(bundle, 'model_bundle.pkl')
print("Bundle saved cleanly.")
