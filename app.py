from flask import Flask, render_template, redirect, request
import pandas as pd
# import joblib
# Importing necessary libraries
from imblearn.over_sampling import SMOTE
import numpy as np
# import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# import xgboost as xgb
from tensorflow.keras.models import load_model
from lime.lime_tabular import LimeTabularExplainer
import base64
import tensorflow as tf
from io import BytesIO
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
# import mysql.connector
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

##############################
####### INDEX SECTION ########
##############################

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')





##################################
####### RAIN FALL SECTION ########
##################################

@app.route('/rainfall')
def rainfall():
    return render_template('rainfall.html')

@app.route('/view1')
def view1():
    df = pd.read_csv(r'Dataset\weatherAUS.csv')
    df = df.head(1000)
    df_html = df.to_html(classes='table table-striped table-hover', index=False, border=0)  
    return render_template('view1.html', table=df_html)

@app.route('/algo1',methods=['GET', 'POST'])
def algo1():
    df = pd.read_csv(r'Dataset\train_data____1.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Placeholder variables for the results
    accuracy = None
    classification_rep = None
    selected_algo = None  

    if request.method == 'POST':
        selected_algo = request.form.get('algorithm')  

        # Naive Bayes
        if selected_algo == 'Naive Bayes':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # Decision Tree
        elif selected_algo == 'Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # Logistic Regression
        elif selected_algo == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # Random Forest
        elif selected_algo == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # XGBoost
        elif selected_algo == 'XGBoost':
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # ANN
        elif selected_algo == 'ANN':
            ann_model = load_model(r'Models\ann_model.h5')  
            y_pred_ann_prob = ann_model.predict(X_test).ravel()
            y_pred_ann = (y_pred_ann_prob > 0.5).astype(int)
            accuracy = round(accuracy_score(y_test, y_pred_ann), 2)
            classification_rep = classification_report(y_test, y_pred_ann, output_dict=True)

        # LSTM
        elif selected_algo == 'LSTM':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

            lstm_model = load_model(r'Models\lstm_rainfall_model.h5')
            y_pred_lstm_prob = lstm_model.predict(X_test_lstm).ravel()
            y_pred_lstm = (y_pred_lstm_prob > 0.5).astype(int)
            accuracy = round(accuracy_score(y_test, y_pred_lstm), 2)
            classification_rep = classification_report(y_test, y_pred_lstm, output_dict=True)
    return render_template('algo1.html', accuracy=accuracy, classification_report=classification_rep, selected_algo=selected_algo)



# # Load dataset for encoding if needed
# df = pd.read_csv(r'Dataset\train_data____1.csv')

# # Initialize LabelEncoder for encoding
# label_encoders = {}
# original_columns = df.select_dtypes(include='object').columns

# # Apply LabelEncoder to each categorical variable
# for col in original_columns:
#     label_encoders[col] = LabelEncoder()
#     df[col] = label_encoders[col].fit_transform(df[col])

# # Splitting the dataset into training and testing sets
# X = df.drop('target', axis=1)
# y = df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Scale the input features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

# # Balancing the training data using SMOTE
# smote = SMOTE(random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# # Load your saved ANN model
# ann_model = tf.keras.models.load_model(r"Models\ann_model.h5")


# # LIME Explanation Integration
# def explain_prediction(user_input, feature_names):
#     """
#     Explains the prediction of the ANN model using LIME.
#     """
#     # Define a prediction function compatible with LIME
#     def predict_fn(data):
#         scaled_data = scaler.transform(data)
#         # Return probabilities for both classes
#         prob_positive = ann_model.predict(scaled_data).ravel()
#         prob_negative = 1 - prob_positive
#         return np.column_stack((prob_negative, prob_positive))
    
#     # Initialize LIME explainer
#     explainer = LimeTabularExplainer(
#         training_data=X_train,
#         feature_names=feature_names,
#         class_names=["No Rain (0)", "Rain (1)"],
#         mode="classification"
#     )
    
#     # Generate explanation
#     explanation = explainer.explain_instance(
#         data_row=np.array(user_input),
#         predict_fn=predict_fn,
#         num_features=len(feature_names)
#     )
#     return explanation

# # Generate Prediction Probability Graph
# def generate_prediction_graph(prediction_prob):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     categories = ['No Rain (0)', 'Rain (1)']
#     probabilities = [1 - prediction_prob, prediction_prob]
    
#     ax.bar(categories, probabilities)
#     ax.set_ylim(0, 1)
#     ax.set_ylabel('Probability')
#     ax.set_title('Prediction Probabilities')
    
#     # Convert to base64 string for frontend
#     img_io = BytesIO()
#     fig.savefig(img_io, format='png')
#     img_io.seek(0)
#     img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
#     plt.close(fig)
#     return img_base64




# Load the saved model and scaler
rain_model = load_model(r'Models\ann_model.h5')
rain_scaler = joblib.load(r'Models\scaler.pkl')

# Load the dataset to get feature names
rain_data = pd.read_csv(r'Dataset\train_data____1.csv')
rain_feature_columns = rain_data.columns[:-1]  # All columns except the target
rain_X = rain_data[rain_feature_columns].values
rain_y = rain_data['target'].values

# Initialize LIME Explainer
rain_explainer = LimeTabularExplainer(
    rain_X,
    training_labels=rain_y,
    feature_names=rain_feature_columns,
    class_names=['No Rain', 'Rain'],
    mode='classification'
)


# Function for prediction
def predict_rain_ann(user_input):
    rain_input_scaled = rain_scaler.transform(np.array(user_input).reshape(1, -1))
    rain_prediction_prob = rain_model.predict(rain_input_scaled).ravel()[0]
    rain_prediction_label = "Rain (1)" if rain_prediction_prob > 0.5 else "No Rain (0)"
    
    if rain_prediction_label == "Rain (1)":
        rain_suggestion = ("Rain is expected tomorrow. It's advisable to carry an umbrella, avoid outdoor activities if possible, "
                      "and ensure your home is protected from potential water accumulation. Plan ahead for any travel to avoid disruptions.")
    else:
        rain_suggestion = ("No rain is forecasted tomorrow. It’s a good day to engage in outdoor activities. However, stay prepared "
                      "for any sudden weather changes and keep yourself updated with the latest weather reports.")
    
    return rain_prediction_label, rain_prediction_prob, rain_suggestion


@app.route('/prediction1', methods=['GET', 'POST'])
def prediction1():
    if request.method == 'POST':
        # Get user inputs from the form
        rain_user_input = [float(request.form[field]) for field in [
            "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed", 
            "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", 
            "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RISK_MM"]]
        
        # # Make prediction
        # prediction_label, prediction_prob, suggestion = predict_rain_ann(user_input)
        
        # # Generate LIME explanation
        # feature_names = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", 
        #                  "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", 
        #                  "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RISK_MM"]
        # explanation = explain_prediction(user_input, feature_names)
        
        # # Convert LIME explanation to an image
        # fig = explanation.as_pyplot_figure()
        # img_io = BytesIO()
        # fig.savefig(img_io, format='png')
        # img_io.seek(0)
        # img_base64_lime = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # # Generate Prediction Probability Graph as base64 image
        # img_base64_prob = generate_prediction_graph(prediction_prob)

        # # Render the template with the results
        # return render_template('prediction1.html', prediction=prediction_label, probability=prediction_prob, suggestion=suggestion, lime_image=img_base64_lime, prob_image=img_base64_prob)
        
        
        rain_user_input = np.array(rain_user_input).reshape(1, -1)

        # Scale the user input
        rain_user_input_scaled = rain_scaler.transform(rain_user_input)

        # Define a wrapper for LIME to handle binary classification
        def predict_proba_for_lime(input_data):
            rain_probabilities = rain_model.predict(input_data)  # Get probabilities for "Rain"
            return np.hstack((1 - rain_probabilities, rain_probabilities))  # Add "No Rain" probabilities

        # Generate LIME explanation
        rain_explanation = rain_explainer.explain_instance(
            rain_user_input_scaled[0],
            predict_proba_for_lime,
            num_features=5
        )

        # Save the explanation plot
        rain_explanation.save_to_file('static/explanation.html')
        plt.figure()
        rain_explanation.as_pyplot_figure()
        plt.savefig('static/explanation.png')

        rain_prediction_label, rain_prediction_prob, rain_suggestion = predict_rain_ann(rain_user_input)
        print(8888888888888, rain_prediction_label)
        print(8888888888888, rain_prediction_prob)
        print(8888888888888, rain_suggestion)

        return render_template('prediction1.html', 
                               image_path = 'static/explanation.png', 
                               html_path = 'static/explanation.html', 
                               prediction = rain_prediction_label, 
                               probability = rain_prediction_prob, 
                               suggestion = rain_suggestion)
    return render_template('prediction1.html')


print('===================================================================================================================')


#############################
####### CROP SECTION ########
#############################


@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/view2')
def view2():
    df = pd.read_csv(r'Dataset\Crop_recommendation.csv')
    df = df.head(1000)
    df_html = df.to_html(classes='table table-striped table-hover', index=False, border=0)  
    return render_template('view2.html', table=df_html)

@app.route('/algo2',methods=['GET', 'POST'])
def algo2():
    df = pd.read_csv(r'Dataset\Crop_recommendation.csv')
    #label encoding the data.
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    # Store original column names
    original_columns = df.select_dtypes(include='object').columns

    # Initialize LabelEncoder
    label_encoders = {}

    # Apply LabelEncoder to each categorical variable
    for col in original_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Print the mapping between original categories and numerical labels
    for col, encoder in label_encoders.items():
        print(f"Mapping for column '{col}':")
        for label, category in enumerate(encoder.classes_):
            print(f"Label {label}: {category}")
    # Splitting the features (X) and target (y)
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.3, random_state=42)
    accuracy = None
    classification_rep = None
    selected_algo = None
    if request.method == 'POST':
        selected_algo = request.form.get('algorithm')  
        lime_image_path = None  
        
        # Decision Tree
        if selected_algo == 'Decision Tree':
            dt_model = DecisionTreeClassifier(random_state=42)
            dt_model.fit(X_train, y_train)
            y_pred_dt = dt_model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred_dt), 2)
            classification_rep = classification_report(y_test, y_pred_dt, output_dict=True)

            # LIME Explanation
            explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=y.unique(), mode='classification')
            i = np.random.randint(0, X_test.shape[0])  
            exp = explainer.explain_instance(X_test.iloc[i].values, dt_model.predict_proba, num_features=5)
            lime_image_path = "static/lime_dt.png"  
            exp.as_pyplot_figure().savefig(lime_image_path)  
        # Random Forest
        elif selected_algo == 'Random Forest':
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred_rf), 2)
            classification_rep = classification_report(y_test, y_pred_rf, output_dict=True)

            # LIME Explanation
            explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=y.unique(), mode='classification')
            i = np.random.randint(0, X_test.shape[0])  
            exp = explainer.explain_instance(X_test.iloc[i].values, rf_model.predict_proba, num_features=5)
            lime_image_path = "static/lime_rf.png"  
            exp.as_pyplot_figure().savefig(lime_image_path)  

        # AdaBoost
        elif selected_algo == 'AdaBoost':
            ab_model = AdaBoostClassifier(random_state=42)
            ab_model.fit(X_train, y_train)
            y_pred_ab = ab_model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred_ab), 2)
            classification_rep = classification_report(y_test, y_pred_ab, output_dict=True)

            # LIME Explanation
            explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=y.unique(), mode='classification')
            i = np.random.randint(0, X_test.shape[0])  
            exp = explainer.explain_instance(X_test.iloc[i].values, ab_model.predict_proba, num_features=5)
            lime_image_path = "static/lime_ab.png"  
            exp.as_pyplot_figure().savefig(lime_image_path)  

        # XGBoost
        elif selected_algo == 'XGBoost':
            xgb_model = XGBClassifier(random_state=42, use_label_encoder=False)
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred_xgb), 2)
            classification_rep = classification_report(y_test, y_pred_xgb, output_dict=True)

            # LIME Explanation
            explainer = LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=y.unique(), mode='classification')
            i = np.random.randint(0, X_test.shape[0])  # Random instance
            exp = explainer.explain_instance(X_test.iloc[i].values, xgb_model.predict_proba, num_features=5)
            lime_image_path = "static/lime_xgb.png" 
            exp.as_pyplot_figure().savefig(lime_image_path)  

        # Return the results and the path to the LIME explanation image to the frontend
        return render_template('algo2.html', accuracy=accuracy, classification_report=classification_rep, selected_algo=selected_algo, lime_image_path=lime_image_path)
    return render_template('algo2.html')
 


# Load dataset
df = pd.read_csv(r'Dataset\Crop_recommendation.csv')

# Label encoding the categorical columns
from sklearn.preprocessing import LabelEncoder

# Store original column names
original_columns = df.select_dtypes(include='object').columns
label_encoders = {}
for col in original_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Splitting features and target
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False)
xgb_model.fit(X_train, y_train)

# Define the crop dictionary
crop_dict = {
    0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
    5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidneybeans',
    10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Moongbeans',
    15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'Pigeonpeas', 19: 'Pomegranate',
    20: 'Rice', 21: 'Watermelon'
}

# Prediction function
def predict_with_xgboost(N, P, K, temperature, humidity, ph, rainfall):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = xgb_model.predict(input_features)
    predicted_crop = crop_dict[prediction[0]]
    return predicted_crop

# LIME Explanation function
def explain_with_lime(input_features):
    explainer = LimeTabularExplainer(X_train.values, training_labels=y_train, mode="classification", 
                                     feature_names=X_train.columns, class_names=np.unique(y_train), discretize_continuous=True)
    exp = explainer.explain_instance(input_features[0], xgb_model.predict_proba, num_features=5)
    return exp

@app.route('/prediction2', methods=['GET', 'POST'])
def prediction2():
    if request.method == 'POST':
        # Get input values from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make the prediction
        predicted_crop = predict_with_xgboost(N, P, K, temperature, humidity, ph, rainfall)

        # Prepare input features for LIME
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Explain the prediction using LIME
        exp = explain_with_lime(input_features)
        
        # Generate a dynamic file name for LIME explanation figure
        lime_image_path = 'static/lime_explanation.png'
        lime_plot = exp.as_pyplot_figure()

        # Save the LIME explanation figure to a file (for displaying in the frontend)
        lime_plot.savefig(lime_image_path)

        return render_template('prediction2.html', predicted_crop=predicted_crop, lime_explanation=lime_image_path)

    return render_template('prediction2.html')

print('===================================================================================================================')




##############################
####### YIELD SECTION ########
##############################


@app.route('/yieldproduction')
def yieldproduction():
    return render_template('yieldproduction.html')

@app.route('/view3')
def view3():
    yield_df2 = pd.read_csv(r'Dataset\Yield Forecasting Data.csv')
    yield_df2 = yield_df2.head(1000)
    yield_df_html2 = yield_df2.to_html(classes='table table-striped table-hover', index=False, border=0)  
    return render_template('view3.html', table=yield_df_html2)



# Load the dataset
yield_df = pd.read_csv(r"Dataset\Yield Forecasting Data.csv")

# Label encoding the data
# Store original column names for categorical data
yield_original_columns = yield_df.select_dtypes(include='object').columns

# Initialize LabelEncoder
yield_label_encoders = {}

# Apply LabelEncoder to each categorical variable
for col in yield_original_columns:
    yield_label_encoders[col] = LabelEncoder()
    yield_df[col] = yield_label_encoders[col].fit_transform(yield_df[col])

# Selecting relevant columns
yield_col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
yield_df = yield_df[yield_col]

# Splitting into features and target
yield_X = yield_df.drop('hg/ha_yield', axis=1)
yield_y = yield_df['hg/ha_yield']

# Scaling the features
yield_scaler = StandardScaler()
yield_X_scaled = yield_scaler.fit_transform(yield_X)

# Split the data into train and test sets
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X_scaled, yield_y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
yield_random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_random_forest_model.fit(yield_X_train, yield_y_train)

@app.route('/prediction3', methods=['GET', 'POST'])
def prediction3():
    if request.method == 'POST':
        # Get input values from the form
        year = float(request.form['year'])
        average_rainfall = float(request.form['average_rainfall'])
        pesticides = float(request.form['pesticides'])
        avg_temp = float(request.form['avg_temp'])
        area = float(request.form['area'])
        item = float(request.form['item'])

        # Prepare the input data as a list (match the order used in the model)
        yield_input_data = [year, average_rainfall, pesticides, avg_temp, area, item]

        # Scale the input data using the same scaler
        yield_inputs_scaled = yield_scaler.transform(np.array(yield_input_data).reshape(1, -1))

        # Predict the yield using the trained linear model
        yield_predicted_yield = yield_random_forest_model.predict(yield_inputs_scaled)

        # Return the predicted yield to the template
        return render_template('prediction3.html', predicted_yield = yield_predicted_yield[0])
    return render_template('prediction3.html')



if __name__ == '__main__':
    app.run(debug=True)
