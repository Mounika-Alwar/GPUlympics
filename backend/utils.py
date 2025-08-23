import pandas as pd

def map_model_size(X:pd.DataFrame)->pd.DataFrame:
    size_map = {
        "7B":0,
        "70B":1,
        "405B":2
    }
    X = X.copy()
    X['model_size'] = X['model_size'].map(size_map)
    return X

def evaluate_gpus(model, preprocessor, model_size, batch_size, learning_rate, seq_length, run_id):
    gpu_types = ['A100','H100','GB200']
    user_input = {
        'model_size':model_size,
        'batch_size':batch_size,
        'learning_rate':learning_rate,
        'seq_length':seq_length,
        'run_id':run_id
    }
    results = []
    for gpu in gpu_types:
        df_input = pd.DataFrame([user_input])
        df_input.insert(0,'gpu_type',gpu)
        X_transformed = preprocessor.transform(df_input)
        pred = model.predict(X_transformed)
        results.append({
            'gpu_type':gpu,
            'training_time_hrs':round(pred[0][0],2),
            'energy_kwh':round(pred[0][1],2),
            'efficiency_tok_per_watt':round(pred[0][2],2)
        })
    fastest_gpu = min(results,key=lambda x:x['training_time_hrs'])
    greenest_gpu = min(results,key=lambda x:x['energy_kwh'])
    efficient_gpu = max(results,key=lambda x:x['efficiency_tok_per_watt'])

    best_gpus = {
        "fastest":fastest_gpu['gpu_type'],
        "greenest":greenest_gpu['gpu_type'],
        "most_efficient":efficient_gpu['gpu_type']
    }

    return results, best_gpus