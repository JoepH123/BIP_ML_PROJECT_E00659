import pandas as pd
from joblib import load
import numpy as np

categorical_features = ["Tdoc", "Iva"]
numerical_features = ['Ateco', 'Importo', 'Conto', 'ContoStd', 'CoDitta', 'TIva', 'Caus']

def show_correct_data(full_dict):
    global categorical_features
    global numerical_features
    
    new_dict = {key: full_dict[key] for key in full_dict if key in categorical_features + numerical_features}
    print(new_dict)
    return [new_dict] #[new_dict]  # output as list, of dictionaries, because thats correct format 


def run_prediction_process(input_data):
    global categorical_features
    global numerical_features

    scaler = load("assets/ui_demo/numerical_scaler.joblib")
    categorical_encoder = load("assets/ui_demo/categorical_encoder.joblib")
    target_variable_encoder = load("assets/ui_demo/target_variable_encoder.joblib")
    dt_model = load("assets/ui_demo/final_decision_tree.joblib")

    df = pd.DataFrame(input_data)
    cat_df = df[categorical_features]
    num_df = df[numerical_features]

    encoded_data = categorical_encoder.transform(cat_df).toarray()
    encoded_df_cat = pd.DataFrame(encoded_data, columns=categorical_encoder.get_feature_names_out(categorical_features))

    scaled_data = scaler.transform(num_df)
    encoded_df_num = pd.DataFrame(scaled_data, columns=scaler.get_feature_names_out(numerical_features))

    total_encoded_df_X = pd.concat([encoded_df_cat, encoded_df_num], axis=1)

    prediction = dt_model.predict(total_encoded_df_X)
    max_index = np.argmax(prediction)
    output_names_target = target_variable_encoder.get_feature_names_out()
    output_name = output_names_target[max_index]
    ivam_code = int(float(output_name.split("_")[1]))

    return ivam_code

