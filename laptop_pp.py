import pickle
import pandas as pd
import re
from flask import Flask, request, jsonify
from sklearn.base import BaseEstimator, TransformerMixin


class ProcessorFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Step 1: Clean text
        df['Processor_Name'] = df['Processor_Name'].str.lower()
        df['Processor_Name'] = df['Processor_Name'].str.replace(r'\s+', ' ', regex=True)
        df['Processor_Name'] = df['Processor_Name'].str.replace(r'\s*processor$', '', regex=True)
        df['Processor_Name'] = df['Processor_Name'].str.strip()

        # Step 2: CPU_Brand
        df['CPU_Brand'] = df['Processor_Name'].apply(
            lambda x: 'intel' if 'intel' in x else (
                'amd' if 'amd' in x else (
                    'apple' if 'apple' in x else 'other'
                )
            )
        )

        # Step 3: CPU_Series
        def extract_series(proc):
            match = re.search(r'(core\s*i\d|ryzen\s*\d|pentium|celeron|athlon|m\d|snapdragon|mediatek)', proc)
            return match.group(0) if match else 'other'

        df['CPU_Series'] = df['Processor_Name'].apply(extract_series)

        # Step 4: CPU_Gen
        df['CPU_Gen'] = df['Processor_Name'].str.extract(r'\((\d+)(?:st|nd|rd|th)\s*gen\)', expand=False)
        df['CPU_Gen'] = pd.to_numeric(df['CPU_Gen'], errors='coerce').fillna(0).astype(int)

        # Step 5: CPU_Cores
        def extract_core_count(proc):
            proc = proc.lower()
            if 'dual-core' in proc: return 2
            if 'quad-core' in proc: return 4
            if 'hexa-core' in proc: return 6
            if 'octa-core' in proc: return 8
            if 'hexadeca-core' in proc: return 16
            return 0

        df['CPU_Cores'] = df['Processor_Name'].apply(extract_core_count)

        # Step 6: Apple chip
        df['Is_Apple_Chip'] = df['Processor_Name'].str.contains('apple', case=False).astype(int)
        df['Apple_Chip_Type'] = df['Processor_Name'].str.extract(r'(m1|m2|max|pro)', expand=False).fillna('none')

        # Return only new columns
        return df[['CPU_Brand', 'CPU_Series', 'CPU_Gen', 'CPU_Cores', 'Is_Apple_Chip', 'Apple_Chip_Type']]
    
class CleanRam(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Use X.iloc[:, 0] to get the first (and only) column as a Series
        return X.iloc[:, 0].str.replace('RAM', '', regex=False)\
                           .str.replace('LP', '', regex=False)\
                           .str.replace('GB', '', regex=False)\
                           .str.extract('(\d+)').astype(float)

class GPUFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['GPU_MEMORY'] = df['GPU'].str.extract(r'(\d+)\s*GB', expand=False).fillna(0).astype(int)
        return df[['GPU_MEMORY','GPU']]

class RAMExpandableExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure input is always a Series (single column)
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        else:
            series = X  # already a Series

        extracted = series.astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)
        return pd.DataFrame(extracted)

class RAMTypeCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle input as a Series (whether X is a DataFrame or a string)
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        else:
            series = pd.Series([X]) if isinstance(X, str) else X

        cleaned = (
            series.astype(str)
            .str.replace(r'\s*RAM$', '', regex=True)
            .str.strip()
        )

        return pd.DataFrame(cleaned)

class GhzExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Always operate on the first column as a Series
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        else:
            series = pd.Series(X)

        # Extract numeric part (e.g., 4.2 from "4.2 Ghz")
        ghz = (
            series.astype(str)
            .str.extract(r'([\d.]+)')
            .fillna(0)
            .astype(float)
        )

        return pd.DataFrame(ghz)

class GPUSeriesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.top_series = None

    def fit(self, X, y=None):
        # Always extract from first column
        series = X.iloc[:, 0].astype(str).str.extract(
            r'(iris xe|uhd \d*|uhd|radeon(?: [a-z0-9]+)*|geforce rtx \d{3,4}|geforce gtx \d{3,4}|geforce mx\d+|hd \d+|integrated)',
            flags=re.IGNORECASE,
            expand=False
        ).fillna('other').str.lower().str.strip()

        self.top_series = series.value_counts().head(self.top_n).index.tolist()
        return self

    def transform(self, X):
        series = X.iloc[:, 0].astype(str).str.extract(
            r'(iris xe|uhd \d*|uhd|radeon(?: [a-z0-9]+)*|geforce rtx \d{3,4}|geforce gtx \d{3,4}|geforce mx\d+|hd \d+|integrated)',
            flags=re.IGNORECASE,
            expand=False
        ).fillna('other').str.lower().str.strip()

        simplified = series.apply(lambda x: x if x in self.top_series else 'other')
        return simplified.to_frame()

class StorageSizeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def convert(value):
            value = str(value).lower()

            if 'tb' in value:
                num = re.search(r'(\d+(\.\d+)?)', value)
                return int(float(num.group(0)) * 1024) if num else 0
            elif 'gb' in value:
                num = re.search(r'(\d+)', value)
                return int(num.group(0)) if num else 0
            elif 'no' in value or value.strip() in ['none', 'nan', 'null']:
                return 0
            elif value.isdigit():
                return int(value)
            else:
                return 0

        # Ensure it's a DataFrame
        if isinstance(X, pd.Series):
            X = X.to_frame()

        return X.applymap(convert)


app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# Load the models
preprocessor = pickle.load(open('./models/preprocessor','rb'))
cat_en = pickle.load(open('./models/cat_en','rb'))
model = pickle.load(open('./models/model','rb'))



@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        input_features = request.json.get("features")

        # Define the expected column order (must exactly match model training)
        column_names = [
            "Brand", "Processor_Name", "RAM_Expandable", "RAM", "RAM_TYPE", "Ghz",
            "Display_type", "Display", "GPU", "GPU_Brand", "SSD", "HDD", "Adapter"
        ]

        # Create a DataFrame from the input
        if not input_features or len(input_features) != len(column_names):
            return jsonify({"error": f"Expected {len(column_names)} features, got {len(input_features)}"}), 400

        df = pd.DataFrame([input_features], columns=column_names)

        # Run preprocessing and prediction
        df_prep = preprocessor.transform(df)
        df_enc = cat_en.transform(pd.DataFrame(df_prep))

        pred = model.predict(df_enc)

        return jsonify({"predicted_price": round(pred[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
