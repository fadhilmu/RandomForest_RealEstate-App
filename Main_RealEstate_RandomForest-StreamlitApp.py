import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, random_feature=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_feature = random_feature
        self.tree = None
        self.feature_importances = None  # Optional

    def fit(self, X, y):
        self.feature_importances = np.zeros(X.shape[1])
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_targets = np.unique(y)

        if len(unique_targets) == 1:
            return unique_targets[0]
        if num_samples < self.min_samples_split:
            return np.mean(y) if len(y) > 0 else 0
        if self.max_depth is not None and depth >= self.max_depth:
            return np.mean(y) if len(y) > 0 else 0

        num_random_features = min(self.random_feature, num_features)

        feature_indices = np.random.choice(num_features, num_random_features, replace=False)

        best_split = self._best_split(X, y, feature_indices)
        if best_split is None:
            return np.mean(y) if len(y) > 0 else 0

        left_mask, right_mask = self._split_data(X[:, best_split['feature']], best_split['value'])

        left_target, right_target = y[left_mask], y[right_mask]
        mse_before = self._calculate_mse(y, y)
        mse_after = self._calculate_mse(left_target, right_target)
        reduction_in_mse = mse_before - mse_after

        self.feature_importances[best_split['feature']] += reduction_in_mse

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': best_split['feature'],
            'value': best_split['value'],
            'left': left_tree,
            'right': right_tree
        }

    def _best_split(self, X, y, feature_indices):
        best_mse = float('inf')
        best_split = None

        for feature in feature_indices:
            values = np.unique(X[:, feature])
            for value in values:
                left_mask, right_mask = self._split_data(X[:, feature], value)

                if len(left_mask) == 0 or len(right_mask) == 0:
                    continue

                mse = self._calculate_mse(y[left_mask], y[right_mask])
                if mse < best_mse:
                    best_mse = mse
                    best_split = {'feature': feature, 'value': value}

        return best_split

    def _split_data(self, feature_column, value):
        left_mask = feature_column <= value
        right_mask = ~left_mask
        return left_mask, right_mask

    def _calculate_mse(self, left_target, right_target):
        if len(left_target) == 0 or len(right_target) == 0:
            return float('inf')
        left_mse = np.mean((left_target - np.mean(left_target)) ** 2) if len(left_target) > 0 else 0
        right_mse = np.mean((right_target - np.mean(right_target)) ** 2) if len(right_target) > 0 else 0
        return (len(left_target) * left_mse + len(right_target) * right_mse) / (len(left_target) + len(right_target))

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['feature']] <= tree['value']:
                return self._predict_single(x, tree['left'])
            else:
                return self._predict_single(x, tree['right'])
        else:
            return tree

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_feature=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_feature = random_feature
        self.trees = []
        self.feature_importances_ = None  # Optional

    def fit(self, X, y):
        self.trees = []
        feature_importances = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):

            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_feature=self.random_feature
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

            feature_importances += tree.feature_importances
        self.feature_importances_ = feature_importances / self.n_estimators

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
    
def TrainTest_Split(X, y, test_size=0.4, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]

    return X_train, X_test, y_train, y_test

def Plot_Price(df):
    df_subset = df.head(1000)
    fig1 = px.scatter(df_subset, x='brokered_by', y='price', 
                  title="Price vs. Brokered By (1000 Sampel)", 
                  labels={'brokered_by': 'Brokered By', 'price': 'Price'},
                  color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig1)

    st.markdown("---")
    
    fig2 = px.scatter(df, x='bed', y='price', 
                      title="Price vs. Bedrooms", 
                      labels={'bed': 'Bedrooms', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig2)

    st.markdown("---")
    
    fig3 = px.scatter(df, x='bath', y='price', 
                      title="Price vs. Bathrooms", 
                      labels={'bath': 'Bathrooms', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig3)

    st.markdown("---")

    fig4 = px.scatter(df, x='acre_lot', y='price', 
                      title="Price vs. Lot Size", 
                      labels={'acre_lot': 'Lot Size (Acres)', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig4)

    st.markdown("---")

    fig5 = px.scatter(df, x='street', y='price', 
                      title="Price vs. Street", 
                      labels={'street': 'Street', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig5)

    st.markdown("---")

    fig6 = px.scatter(df, x='zip_code', y='price', 
                      title="Price vs. Zip Code", 
                      labels={'zip_code': 'Zip Code', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig6)

    st.markdown("---")

    fig7 = px.scatter(df, x='house_size', y='price', 
                      title="Price vs. House Size", 
                      labels={'house_size': 'House Size', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig7)

    st.markdown("---")

    fig8 = px.scatter(df, x='city_encoded', y='price', 
                      title="Price vs. City", 
                      labels={'city_encoded': 'City', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig8)

    st.markdown("---")

    fig9 = px.scatter(df, x='state_encoded', y='price',
                      title="Price vs. State",
                      labels={'state_encoded': 'State', 'price': 'Price'},
                      color='price', color_continuous_scale='Viridis')
    st.plotly_chart(fig9)

    st.markdown("---")

def Prediction():
    st.title("Prediksi Harga Real Estate")

    model_option = st.selectbox("Pilih Model:", ("Gunakan Model Bawaan", "Unggah Model"))

    if model_option == "Gunakan Model Bawaan":
        try:
            default_model_path = "RandomForest_Model.joblib"
            loaded_rf = joblib.load(default_model_path)
            st.success("Model Bawaan telah Dimuat!")
        except Exception as e:
            st.error(f"Error loading default model: {e}")

    elif model_option == "Unggah Model":
        uploaded_model = st.file_uploader("Unggah Model RandomForest untuk Real Estate", type=["joblib"])
        if uploaded_model is not None:
            loaded_rf = joblib.load(uploaded_model)
            st.success("Model Sukses Dimuat!")

    if 'loaded_rf' not in locals():
        st.warning("Mohon Pilih Model.")
        return

    dataset = pd.read_csv("Realtor50K.csv")
    st.write(dataset.head())

    data = dataset.drop(columns=["city", "state", "status", "prev_sold_date"])

    X = data[['brokered_by', 'bed', 'bath', 'acre_lot', 'street', 'zip_code', 'house_size', 'city_encoded', 'state_encoded']].values
    y = data['price'].values

    # X_train, X_test, y_train, y_test = TrainTest_Split(X, y, test_size=0.2) # apabila ingin acak terus
    X_train, X_test, y_train, y_test = TrainTest_Split(X, y, test_size=0.2, random_state=42)

    st.write("\nSeluruh Data:", X.shape[0])
    st.write("\n_Training Set:_", X_train.shape[0])
    st.write("_Test Set:_", X_test.shape[0])

    st.subheader("Prediksi Input Manual")

    states = dataset['state'].unique()
    cities = dataset['city'].unique()
    brokers = dataset['brokered_by'].unique()
    streets = dataset['street'].unique()
    zip_codes = dataset['zip_code'].unique()

    brokered_by = st.selectbox("Pilih Broker", options=brokers, index=0)
    col1, col2 = st.columns(2)
    with col1:
        bed = st.number_input("Masukkan Jumlah Kamar Tidur **(1 - 10)**", min_value=1, max_value=10, value=7)
        bath = st.number_input("Masukkan Jumlah Kamar Mandi **(1 - 10)**", min_value=1, max_value=10, value=3)
        acre_lot = st.number_input("Masukkan Ukuran Tanah **(0.01 - 10.0)**", min_value=0.01, max_value=10.0, value=0.09)
        street = st.selectbox("Pilih Jalan", options=streets, index=0)

    with col2:
        zip_code = st.selectbox("Pilih Kode Pos", options=zip_codes, index=0)
        house_size = st.number_input("Masukkan Ukuran Rumah **(500 - 10000 sqft)**", min_value=500, max_value=10000, value=1192)
        city = st.selectbox("Pilih Kota", options=sorted(cities), index=0)
        state = st.selectbox("Pilih Negara", options=sorted(states), index=0)

    state_mapping = {state: idx + 1 for idx, state in enumerate(states)}
    state_encoded = state_mapping[state]

    city_mapping = {city: idx + 1 for idx, city in enumerate(cities)}
    city_encoded = city_mapping[city]

    input_data = np.array([[brokered_by, bed, bath, acre_lot, street, zip_code, house_size, city_encoded, state_encoded]])

    if st.button("**Predict Price**"):
        prediction = loaded_rf.predict(input_data)
        st.write(f"Prediksi Harga: ${prediction[0]:,.2f}\n\n")
        st.write("\n")
        st.write("\n")
    
    st.markdown("---")

    y_test_pred = loaded_rf.predict(X_test)
    y_test_pred_rounded = np.round(y_test_pred, 2)

    comparison_test = pd.DataFrame({
        'True Price': y_test,
        'Predicted Price': y_test_pred_rounded
    })

    st.write("\n")
    st.write("Perbandingan Sampel (Harga Asli vs Prediksi terhadap _Test Set):_")
    st.write(comparison_test.head(10))

    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    st.write("\nMetriks Evaluasi Model terhadap _Test Set:_")
    st.write(f"R² Score: {r2_test:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse_test:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae_test:.4f}")
    st.write("\n")
    st.write("\n")

    st.markdown("---")

    y_train_pred = loaded_rf.predict(X_train)
    y_train_pred_rounded = np.round(y_train_pred, 2)

    comparison_train = pd.DataFrame({
        'True Price': y_train,
        'Predicted Price': y_train_pred_rounded
    })

    st.write("\nPerbandingan Sampel (Harga Asli vs Prediksi terhadap _Training Set):_")
    st.write(comparison_train.head(10))

    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)

    st.write("\nMetriks Evaluasi Model terhadap _Training Set:_")
    st.write(f"R² Score: {r2_train:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse_train:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae_train:.4f}")
    st.write("\n")
    st.write("\n")

    st.markdown("---")

    y_all_pred = loaded_rf.predict(X)
    y_all_pred_rounded = np.round(y_all_pred, 2)

    comparison_all = pd.DataFrame({
        'True Price': y,
        'Predicted Price': y_all_pred_rounded
    })

    st.write("\nPerbadingan Seluruh Data (Harga Asli vs Prediksi):")
    st.write(comparison_all.head(10))

    r2_all = r2_score(y, y_all_pred)
    mse_all = mean_squared_error(y, y_all_pred)
    mae_all = mean_absolute_error(y, y_all_pred)

    st.write("\nMetriks Evaluasi Model terhadap Seluruh Data:")
    st.write(f"R² Score: {r2_all:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse_all:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae_all:.4f}")

    st.title("Price vs Features Scatter Plots")
    Plot_Price(data)

    feature_names = ['brokered_by', 'bed', 'bath', 'acre_lot', 'street', 'zip_code', 'house_size', 'city_encoded', 'state_encoded']
    feature_importances = loaded_rf.feature_importances_ 
    normalized_importances = feature_importances / feature_importances.sum()

    st.write("### Feature Importances")
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.array(normalized_importances) * 100
    })
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', 
                title="Feature Importances", 
                labels={'Importance': 'Feature Importance (%)', 'Feature': 'Features'},
                color='Importance', color_continuous_scale='Blues')
    fig.update_traces(
        hovertemplate='Feature: %{y}<br>Importance: %{x:.2f}%<extra></extra>'
    )
    st.plotly_chart(fig)

def Train():
    data = pd.read_csv("Realtor5K.csv")

    st.write(f"Jumlah Data: {len(data)}")

    st.subheader("Dataset Sampel")
    st.write(data.head())

    default_target = 'price'

    st.write(f"Target (y): {default_target}")

    feature_columns = st.multiselect(
        'Pilih Fitur (X)', 
        options=[col for col in data.columns if col != default_target],
        default=['brokered_by', 'bed', 'bath', 'acre_lot', 'street', 'zip_code', 'house_size', 'city_encoded', 'state_encoded']
    )

    X = data[feature_columns].values
    y = data[default_target].values
    st.write(f"Fitur Pilihan: {feature_columns}")

    test_sizes = [10, 20, 30, 40, 50]
    test_size = st.select_slider("Test Size (%)", options=test_sizes, value=20) / 100

    X_train, X_test, y_train, y_test = TrainTest_Split(X, y, test_size=test_size, random_state=42)

    st.write(f"Porsi Tes: {test_size * 100}%")
    st.write(f"Jumlah _training data:_ {len(X_train)} samples")
    st.write(f"Jumlah _test data:_ {len(X_test)} samples")

    n_estimators = st.number_input("Jumlah Pohon Keputusan", min_value=5, max_value=20, value=10, step=5)
    max_depth = st.number_input("Max Depth", min_value=0, max_value=20, value=0, step=1)
    if max_depth == 0:
        max_depth = None
    if max_depth is None:
        st.write("Kedalaman Node Maksimal = None (sampai paling optimal)")
    else:
        st.write(f"Kedalaman Node Maksimal = {max_depth}")

    fractions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    random_feature = st.select_slider("Random Feature Fraction", options=fractions, value=3)
    num_features = X.shape[1]
    st.write(f"Jumlah Seleksi Fitur secara Acak: **{random_feature}** dari {num_features} Total Fitur per _Split_")

    st.markdown("---")

    st.write(f"NOTE: apabila Anda menggunakan Nilai Default, maka Training Model memerlukan Waktu 20 Detik")

    if st.button("**Train Model**"):
        if not feature_columns:
            st.warning("Mohon pilih minimal satu fitur!")
        else:
            st.balloons()
            rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_feature=random_feature)
            rf.fit(X_train, y_train)

            y_test_pred = rf.predict(X_test)
            y_test_pred_rounded = np.round(y_test_pred, 0)

            comparison_test = pd.DataFrame({
                'True Price': y_test,
                'Predicted Price': y_test_pred_rounded
            })

            st.subheader("Harga Asli vs Prediksi (_Test Set_)")
            st.write(comparison_test.head(10))

            r2_test = r2_score(y_test, y_test_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)

            st.subheader("Metriks Evaluasi Model")
            st.write(f"R² Score: {r2_test:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse_test:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae_test:.4f}")

            model_filename = "RandomForest_SampleModel"  
            st.write(f"The model will be saved as: **{model_filename}.joblib**")
            model_filepath = f"{model_filename}.joblib"

            if model_filename:
                try:
                    model_buffer = io.BytesIO()
                    joblib.dump(rf, model_buffer)           
                    model_buffer.seek(0)
                                
                    download_button_clicked = st.download_button(
                        label="Download Model",
                        data=model_buffer,
                        file_name=model_filepath,
                        mime="application/octet-stream"
                    )
                    if download_button_clicked:                
                        st.success(f"Model '{model_filename}' has been successfully saved and is ready for download!")
                except Exception as e:
                    st.error(f"Error occurred while saving the model: {e}")

def Visualize():
    class DecisionTreeVisual:
        def __init__(self, max_depth=None, min_samples_split=2, random_feature_fraction=0.5, feature_names=None):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.random_feature_fraction = random_feature_fraction
            self.tree = None
            self.tree_id = None 
            self.feature_names = feature_names 

        def fit(self, X, y, tree_id=None):
            self.tree_id = tree_id
            self.tree = self._build_tree(X, y)

        def _build_tree(self, X, y, depth=0):
            num_samples, num_features = X.shape
            unique_targets = np.unique(y)

            if len(unique_targets) == 1:
                return unique_targets[0]
            if num_samples < self.min_samples_split:
                return np.mean(y) if len(y) > 0 else 0
            if self.max_depth is not None and depth >= self.max_depth:
                return np.mean(y) if len(y) > 0 else 0

            feature_indices = np.random.choice(num_features, int(self.random_feature_fraction * num_features), replace=False)

            best_split = self._best_split(X, y, feature_indices)
            if best_split is None:
                return np.mean(y) if len(y) > 0 else 0

            left_mask, right_mask = self._split_data(X[:, best_split['feature']], best_split['value'])

            left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

            return {
                'feature': best_split['feature'],
                'value': best_split['value'],
                'left': left_tree,
                'right': right_tree,
                'left_samples': len(y[left_mask]),
                'right_samples': len(y[right_mask])
            }

        def _best_split(self, X, y, feature_indices):
            best_mse = float('inf')
            best_split = None

            for feature in feature_indices:
                feature_name = self.feature_names[feature] if self.feature_names is not None else f"Feature {feature}"            
                values = np.unique(X[:, feature])
                for value in values:
                    left_mask, right_mask = self._split_data(X[:, feature], value)

                    if len(left_mask) == 0 or len(right_mask) == 0:
                        continue

                    mse = self._calculate_mse(y[left_mask], y[right_mask])

                    st.write(f"Evaluating split at feature '{feature_name}', value {value}: MSE = {mse:.4f}")

                    if mse < best_mse:
                        best_mse = mse
                        best_split = {'feature': feature, 'value': value}

            if best_split:
                best_feature_name = self.feature_names[best_split['feature']] if self.feature_names is not None else f"Feature {best_split['feature']}"
                st.write(f"Tree {self.tree_id} - Best split found: Feature '{best_feature_name}' with value {best_split['value']} (MSE = {best_mse:.4f})")
                st.markdown("---")
            return best_split

        def _split_data(self, feature_column, value):
            left_mask = feature_column <= value
            right_mask = ~left_mask
            return left_mask, right_mask

        def _calculate_mse(self, left_target, right_target):
            if len(left_target) == 0 or len(right_target) == 0:
                return float('inf')
            left_mse = np.mean((left_target - np.mean(left_target)) ** 2) if len(left_target) > 0 else 0
            right_mse = np.mean((right_target - np.mean(right_target)) ** 2) if len(right_target) > 0 else 0
            return (len(left_target) * left_mse + len(right_target) * right_mse) / (len(left_target) + len(right_target))

        def _print_tree_recursive(self, node, depth=0):
            indent = "&nbsp;" * (depth * 4)
            if isinstance(node, dict):
                st.markdown(f"{indent}Feature '{self.feature_names[node['feature']]}', Value: {node['value']}")
                st.markdown(f"{indent}&nbsp;&nbsp;Left: {node['left_samples']} samples")
                self._print_tree_recursive(node['left'], depth + 1)
                st.markdown(f"{indent}&nbsp;&nbsp;Right: {node['right_samples']} samples")
                self._print_tree_recursive(node['right'], depth + 1)
            else:
                st.markdown(f"{indent}Leaf: {node}")

        def print_tree(self):
            """ Tree Visualization """
            self._print_tree_recursive(self.tree)

    class RandomForestVisual:
        def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_feature_fraction=0.5, feature_names=None):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.random_feature_fraction = random_feature_fraction
            self.trees = []
            self.feature_names = feature_names
        
        def __str__(self):
            return (f"Random Forest with {self.n_estimators} estimators, "
                    f"max depth = {self.max_depth}, "
                    f"min samples split = {self.min_samples_split}, "
                    f"random feature fraction = {self.random_feature_fraction:.2f}. "
                    f"Feature names: {self.feature_names}")
        
        def fit(self, X, y):
            self.trees = []
            for tree_id in range(1, self.n_estimators + 1):
                
                bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]

                sample_counts = {i: list(bootstrap_indices).count(i) for i in range(len(X))}
                st.write(f"\nTree {tree_id} - Bootstrap Sample Details:")
                for idx, count in sample_counts.items():
                    st.write(f"  Sample {idx + 1}: {count} times")

                tree = DecisionTreeVisual(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    random_feature_fraction=self.random_feature_fraction,
                    feature_names=self.feature_names
                )
                tree.fit(X_bootstrap, y_bootstrap, tree_id=tree_id)
                self.trees.append(tree)

        def print_forest(self):
            """ Forest Visualization """
            for i, tree in enumerate(self.trees):
                st.write(f"Tree {i+1}:")
                tree.print_tree()
                st.markdown("---")

    data_test = {
        'price': [
            110000.0, 950000.0, 6899000.0, 525000.0, 289900.0, 
            384900.0, 199999.0, 419000.0, 745000.0, 875000.0
        ],
        'bed': [
            7.0, 5.0, 4.0, 3.0, 3.0, 
            3.0, 3.0, 4.0, 4.0, 4.0
        ],
        'bath': [
            3.0, 4.0, 6.0, 3.0, 2.0, 
            2.0, 2.0, 2.0, 3.0, 4.0
        ],
        'acre_lot': [
            0.09, 0.99, 0.83, 0.45, 0.36, 
            0.46, 1.76, 2.0, 0.56, 1.5
        ],
        'house_size': [
            1192.0, 5000.0, 4600.0, 2314.0, 1276.0, 
            1476.0, 1968.0, 1607.0, 2847.0, 4366.0
        ],
        'state_encoded': [
            0, 1, 1, 2, 2, 
            2, 2, 3, 3, 2
        ],
        'brokered_by': [
            92147.0, 94933.0, 103341.0, 21163.0, 67455.0, 
            97400.0, 33714.0, 22188.0, 13548.0, 59566.0
        ],
        'zip_code': [
            949.0, 802.0, 802.0, 1001.0, 1001.0, 
            1001.0, 1001.0, 1002.0, 1002.0, 1002.0
        ],
        'city_encoded': [
            0, 1, 1, 2, 2, 
            2, 2, 3, 3, 3
        ]
    }

    feature_names = ['brokered_by', 'bed', 'bath', 'acre_lot', 'street', 'zip_code', 'house_size', 'city_encoded', 'state_encoded']

    df = pd.DataFrame(data_test)
    df.index = df.index + 1

    X = df.drop('price', axis=1).values
    y = df['price'].values

    random_state_option = st.selectbox(
        'Pilih Nilai Random State (for fun)',
        ['None', '0', 'Custom'],
        index=2
    )
    if random_state_option == 'None':
        random_state = None
    elif random_state_option == '0':
        random_state = 0
    else:
        random_state = st.slider('Select Random State', min_value=1, max_value=50, value=42, step=1)

    X_train, X_test, y_train, y_test = TrainTest_Split(X, y, test_size=0.2, random_state=random_state)

    st.write(df)
    st.write("\nSeluruh Data:", X.shape[0])
    st.write("\n_Training Set:_", X_train.shape[0])
    st.write('\n')
    st.write('\n')
    st.text("Berikut Pilihan Algoritma Model secara Sederhana.")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_estimators = st.number_input("Jumlah Pohon Keputusan (2-3)", min_value=2, max_value=3, value=2)

    with col2:
        max_depth = st.number_input("Kedalaman Maksimal Node (1-3)", min_value=1, max_value=3, value=2)

    with col3:
        fractions = [0.2, 0.4, 0.6, 0.8, 1]
        random_feature_fraction = st.select_slider("Random Feature Fraction", options=fractions, value=0.4)

    st.markdown("_Refresh_ untuk Perubahan.")
    if st.button('Refresh'):
        st.rerun()
    st.write("\n")

    rf = RandomForestVisual(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=4, random_feature_fraction=random_feature_fraction, feature_names=feature_names)
    st.write(rf)

    rf.fit(X_train, y_train)
    rf.print_forest()

st.set_page_config(page_title="Real Estate Prediction", layout="wide")
st.sidebar.title("Real Estate Prediction App")

menu = ["Prediction Model", "Train Model", "About Model"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

if choice == "Prediction Model":
    Prediction()

elif choice == "Train Model":
    Train()

elif choice == "About Model":
    st.title("About Random Forest Regressor")
    st.text("Sesi ini akan memberikan Visualisasi terhadap Proses Algoritma Beroperasi")
    Visualize()

st.sidebar.write("Developed by Fadhil, Fahmi & Monika")
