import pickle
import joblib
import os

model_path = 'parameters/knn_model.pkl'

print('Trying pickle...')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print('Pickle successful')
    print(f'Model type: {type(model)}')
except Exception as e:
    print(f'Pickle failed: {e}')

print('\nTrying joblib...')
try:
    model = joblib.load(model_path)
    print('Joblib successful')
    print(f'Model type: {type(model)}')
except Exception as e:
    print(f'Joblib failed: {e}')
