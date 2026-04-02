from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import os
from werkzeug.utils import secure_filename
from groq import Groq
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════
#  ★  CONFIGURATION — only section you ever need to edit  ★
# ════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # set in Render dashboard
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

UPLOAD_FOLDER    = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER']   = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER,    exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _use_groq() -> bool:
    """True when a valid Groq key is configured."""
    return bool(GROQ_API_KEY and GROQ_API_KEY.startswith('gsk_'))


# ─── Data Quality Scorer ────────────────────────────────────────────────────
class DataQualityScorer:

    @staticmethod
    def calculate_quality_score(profile: Dict) -> Dict:
        scores = {'overall': 0, 'completeness': 0, 'consistency': 0, 'validity': 0, 'components': {}}

        total_cells   = profile['basic_info']['rows'] * profile['basic_info']['columns']
        missing_cells = profile['missing_values']['total_missing']

        completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        scores['completeness'] = round(completeness, 2)

        dup_penalty      = (profile['basic_info']['duplicate_rows'] / profile['basic_info']['rows'] * 100) if profile['basic_info']['rows'] > 0 else 0
        constant_cols    = sum(1 for c in profile['column_analysis'] if c['is_constant'])
        constant_penalty = (constant_cols / profile['basic_info']['columns'] * 100) if profile['basic_info']['columns'] > 0 else 0
        consistency      = max(0, 100 - dup_penalty - constant_penalty)
        scores['consistency'] = round(consistency, 2)

        id_cols    = sum(1 for c in profile['column_analysis'] if c['is_identifier'])
        id_penalty = (id_cols / profile['basic_info']['columns'] * 50) if profile['basic_info']['columns'] > 0 else 0
        validity   = max(0, 100 - id_penalty)
        scores['validity'] = round(validity, 2)

        scores['overall'] = round(
            scores['completeness'] * 0.4 + scores['consistency'] * 0.3 + scores['validity'] * 0.3, 2
        )
        scores['components'] = {
            'missing_data_impact': round((missing_cells / total_cells * 100) if total_cells > 0 else 0, 2),
            'duplicate_impact':    round(dup_penalty, 2),
            'constant_columns':    constant_cols,
            'identifier_columns':  id_cols,
        }

        if   scores['overall'] >= 80: scores['rating'], scores['color'] = 'Excellent', 'green'
        elif scores['overall'] >= 60: scores['rating'], scores['color'] = 'Good',      'green'
        elif scores['overall'] >= 40: scores['rating'], scores['color'] = 'Fair',      'orange'
        else:                         scores['rating'], scores['color'] = 'Poor',      'red'

        return scores


# ─── Dataset Profiler ───────────────────────────────────────────────────────
class DatasetProfiler:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze(self) -> Dict:
        profile = {
            'basic_info':          self._get_basic_info(),
            'column_analysis':     self._analyze_columns(),
            'missing_values':      self._analyze_missing_values(),
            'data_types':          self._analyze_data_types(),
            'statistical_summary': self._get_statistical_summary(),
            'potential_issues':    self._identify_issues(),
            'ml_problem_type':     self._detect_ml_problem_type(),
        }
        profile['quality_score'] = DataQualityScorer.calculate_quality_score(profile)
        return profile

    def _get_basic_info(self) -> Dict:
        return {
            'rows':           len(self.df),
            'columns':        len(self.df.columns),
            'memory_usage':   f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'duplicate_rows': int(self.df.duplicated().sum()),
            'column_names':   list(self.df.columns),
        }

    def _analyze_columns(self) -> List[Dict]:
        result = []
        for col in self.df.columns:
            info = {
                'name':               col,
                'dtype':              str(self.df[col].dtype),
                'unique_values':      int(self.df[col].nunique()),
                'missing_count':      int(self.df[col].isnull().sum()),
                'missing_percentage': float(self.df[col].isnull().sum() / len(self.df) * 100),
                'is_constant':        self.df[col].nunique() == 1,
                'is_identifier':      self._is_identifier_column(col),
                'sample_values':      [str(v) for v in self.df[col].dropna().head(5).tolist()],
            }
            if pd.api.types.is_numeric_dtype(self.df[col]):
                info['numeric_stats'] = {
                    'mean':  float(self.df[col].mean())  if not self.df[col].isnull().all() else None,
                    'std':   float(self.df[col].std())   if not self.df[col].isnull().all() else None,
                    'min':   float(self.df[col].min())   if not self.df[col].isnull().all() else None,
                    'max':   float(self.df[col].max())   if not self.df[col].isnull().all() else None,
                    'zeros': int((self.df[col] == 0).sum()),
                }
            result.append(info)
        return result

    def _is_identifier_column(self, col: str) -> bool:
        if self.df[col].nunique() == len(self.df):
            return True
        return any(p in col.lower() for p in ['id', 'index', 'key', 'code', 'number'])

    def _analyze_missing_values(self) -> Dict:
        info = {
            'total_missing':        int(self.df.isnull().sum().sum()),
            'columns_with_missing': [],
            'missing_percentage':   float(self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100),
        }
        for col in self.df.columns:
            mc = self.df[col].isnull().sum()
            if mc > 0:
                info['columns_with_missing'].append({
                    'column': col, 'count': int(mc),
                    'percentage': float(mc / len(self.df) * 100),
                })
        return info

    def _analyze_data_types(self) -> Dict:
        tc = {'numeric': 0, 'categorical': 0, 'datetime': 0, 'text': 0}
        for col in self.df.columns:
            if   pd.api.types.is_numeric_dtype(self.df[col]):        tc['numeric']     += 1
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]): tc['datetime']    += 1
            elif self.df[col].nunique() / len(self.df) < 0.05:       tc['categorical'] += 1
            else:                                                      tc['text']        += 1
        return tc

    def _get_statistical_summary(self) -> Dict:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            return {col: {k: float(v) for k, v in stats.items()}
                    for col, stats in self.df[num_cols].describe().to_dict().items()}
        return {}

    def _identify_issues(self) -> List[Dict]:
        issues = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                issues.append({'type': 'constant_column', 'severity': 'high', 'column': col,
                                'description': f'Column "{col}" has only one unique value'})
        for col in self.df.columns:
            mp = self.df[col].isnull().sum() / len(self.df) * 100
            if mp > 50:
                issues.append({'type': 'high_missing_values', 'severity': 'high', 'column': col,
                                'description': f'Column "{col}" has {mp:.1f}% missing values'})
        dc = self.df.duplicated().sum()
        if dc > 0:
            issues.append({'type': 'duplicate_rows', 'severity': 'medium',
                           'description': f'Dataset contains {dc} duplicate rows'})
        for col in self.df.columns:
            if self._is_identifier_column(col):
                issues.append({'type': 'identifier_column', 'severity': 'low', 'column': col,
                                'description': f'Column "{col}" appears to be an identifier'})
        return issues

    def _detect_ml_problem_type(self) -> Dict:
        targets = []
        n_rows = len(self.df)

        for col in self.df.columns:
            if self._is_identifier_column(col):
                continue

            uc = self.df[col].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
            is_categorical = self.df[col].dtype in ['object', 'category']

            # Categorical columns → always classification
            if is_categorical:
                if uc == 2:
                    targets.append({'column': col, 'type': 'binary_classification', 'confidence': 'high', 'unique_values': int(uc)})
                elif uc <= 20:
                    targets.append({'column': col, 'type': 'multiclass_classification', 'confidence': 'high', 'unique_values': int(uc)})
                continue

            if not is_numeric:
                continue

            # Integer columns with few unique values → classification
            is_int = str(self.df[col].dtype).startswith('int') or (
                self.df[col].dropna() == self.df[col].dropna().astype(int)
            ).all() if not self.df[col].isnull().all() else False

            ratio = uc / n_rows  # unique ratio

            if uc == 2:
                targets.append({'column': col, 'type': 'binary_classification', 'confidence': 'high', 'unique_values': int(uc)})
            elif uc <= 20 and (is_int or ratio < 0.01):
                # Few unique integers or very low cardinality → classification
                targets.append({'column': col, 'type': 'multiclass_classification', 'confidence': 'high', 'unique_values': int(uc)})
            elif ratio > 0.05:
                # High cardinality continuous → regression
                targets.append({'column': col, 'type': 'regression', 'confidence': 'medium', 'unique_values': int(uc)})
            else:
                targets.append({'column': col, 'type': 'multiclass_classification', 'confidence': 'low', 'unique_values': int(uc)})

        # Pick best suggested type — prefer classification candidates
        classification_types = [t for t in targets if 'classification' in t['type']]
        regression_types     = [t for t in targets if t['type'] == 'regression']

        if classification_types:
            # Sort by confidence: high > medium > low, fewer unique values first
            classification_types.sort(key=lambda x: ({'high':0,'medium':1,'low':2}[x['confidence']], x['unique_values']))
            suggested = classification_types[0]['type']
        elif regression_types:
            suggested = 'regression'
        else:
            suggested = 'unknown'

        return {'potential_targets': targets, 'suggested_type': suggested}


# ─── Preprocessing Engine ───────────────────────────────────────────────────
class PreprocessingEngine:
    """Groq AI when key is set, rule-based fallback otherwise."""

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if _use_groq() else None

    def generate_recommendations(self, profile: Dict) -> Dict:
        if self.client:
            try:
                return self._groq_recommendations(profile)
            except Exception as e:
                print(f"[Groq error — falling back to rule-based] {e}")
        return self._rule_based_recommendations(profile)

    # ── Groq path ───────────────────────────────────────────────────────────

    def _groq_recommendations(self, profile: Dict) -> Dict:
        cols = '\n'.join(
            f"  - {c['name']} ({c['dtype']}, {c['unique_values']} unique, "
            f"{c['missing_percentage']:.1f}% missing"
            + (" [CONSTANT]" if c['is_constant'] else "")
            + (" [ID-LIKE]"  if c['is_identifier'] else "") + ")"
            for c in profile['column_analysis'][:20]
        )
        prompt = f"""You are a data preprocessing expert. Return preprocessing recommendations as JSON only.

DATASET
  Rows: {profile['basic_info']['rows']} | Columns: {profile['basic_info']['columns']}
  Duplicates: {profile['basic_info']['duplicate_rows']} | Missing: {profile['missing_values']['missing_percentage']:.1f}%
  Quality: {profile['quality_score']['overall']}/100 ({profile['quality_score']['rating']})
  ML task: {profile['ml_problem_type']['suggested_type']}

COLUMNS
{cols}

Return ONLY valid JSON (no markdown, no backticks):
{{"source":"ai_groq","overall_assessment":"<2-3 sentences>","columns_to_drop":[{{"column":"<name>","reason":"<why>"}}],"missing_value_strategy":{{"<col>":"mean|median|mode|drop_rows"}},"encoding_recommendations":{{"<col>":"one_hot_encoding|label_encoding|target_encoding"}},"scaling_recommendation":"standard|minmax|robust|none","outlier_handling":"iqr|zscore|none","data_split":{{"train":0.7,"validation":0.15,"test":0.15}},"preprocessing_steps":["<step>"],"suggestions":[{{"text":"<insight>","severity":"high|medium|low","impact":"<effect>"}}]}}"""

        resp    = self.client.chat.completions.create(
            model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2500)
        ai_text = resp.choices[0].message.content

        import re
        try:
            m   = re.search(r'\{[\s\S]*\}', ai_text)
            rec = json.loads(m.group() if m else ai_text)
            rec['source']        = 'ai_groq'
            rec['quality_score'] = profile['quality_score']
            rb = self._rule_based_recommendations(profile)
            for key in rb:
                rec.setdefault(key, rb[key])
            return rec
        except Exception:
            return self._rule_based_recommendations(profile)

    # ── Rule-based path ─────────────────────────────────────────────────────

    def _rule_based_recommendations(self, profile: Dict) -> Dict:
        rec = {
            'source': 'rule_based', 'quality_score': profile['quality_score'],
            'columns_to_drop': [], 'missing_value_strategy': {},
            'encoding_recommendations': {}, 'scaling_recommendation': 'standard',
            'outlier_handling': 'iqr', 'data_split': {'train': 0.7, 'validation': 0.15, 'test': 0.15},
            'preprocessing_steps': [], 'suggestions': [], 'overall_assessment': '',
        }

        for ci in profile['column_analysis']:
            if ci['is_constant']:
                rec['columns_to_drop'].append({'column': ci['name'], 'reason': 'Constant — carries no information'})
            elif ci['is_identifier']:
                rec['columns_to_drop'].append({'column': ci['name'], 'reason': 'Identifier — not a feature'})
            elif ci['missing_percentage'] > 70:
                rec['columns_to_drop'].append({'column': ci['name'], 'reason': f'{ci["missing_percentage"]:.1f}% missing — too sparse'})

        drop_set = {d['column'] for d in rec['columns_to_drop']}
        for ci in profile['column_analysis']:
            if ci['name'] in drop_set or ci['missing_count'] == 0:
                continue
            if   'numeric_stats' in ci:      rec['missing_value_strategy'][ci['name']] = 'median'
            elif ci['unique_values'] < 10:   rec['missing_value_strategy'][ci['name']] = 'mode'
            else:                             rec['missing_value_strategy'][ci['name']] = 'drop_rows'

        for ci in profile['column_analysis']:
            if ci['name'] in drop_set or ci['dtype'] not in ['object', 'category']:
                continue
            if   ci['unique_values'] == 2:  rec['encoding_recommendations'][ci['name']] = 'label_encoding'
            elif ci['unique_values'] <= 10: rec['encoding_recommendations'][ci['name']] = 'one_hot_encoding'
            else:                            rec['encoding_recommendations'][ci['name']] = 'target_encoding'

        steps = []
        if profile['basic_info']['duplicate_rows'] > 0:
            steps.append(f"Remove {profile['basic_info']['duplicate_rows']} duplicate rows")
        if rec['columns_to_drop']:
            steps.append(f"Drop {len(rec['columns_to_drop'])} irrelevant columns")
        if rec['missing_value_strategy']:
            steps.append(f"Impute missing values in {len(rec['missing_value_strategy'])} columns")
        if rec['encoding_recommendations']:
            steps.append(f"Encode {len(rec['encoding_recommendations'])} categorical columns")
        steps += ["Cap outliers with IQR method", "Scale features with StandardScaler", "Split 70/15/15 train/val/test"]
        rec['preprocessing_steps'] = steps

        qs, mp, dc = profile['quality_score']['overall'], profile['missing_values']['missing_percentage'], profile['basic_info']['duplicate_rows']
        if qs < 60: rec['suggestions'].append({'text': f'Quality {qs}/100 — significant cleanup needed.', 'severity': 'high',   'impact': 'Critical for model performance'})
        if mp > 10: rec['suggestions'].append({'text': f'{mp:.1f}% missing data detected.',                'severity': 'medium', 'impact': 'May reduce accuracy'})
        if dc > 0:  rec['suggestions'].append({'text': f'{dc} duplicate rows will be removed.',            'severity': 'medium', 'impact': 'Prevents data leakage'})
        if not rec['suggestions']:
            rec['suggestions'].append({'text': 'Dataset is clean — standard preprocessing applied.', 'severity': 'low', 'impact': 'Ready for training'})

        rec['overall_assessment'] = (
            f"{profile['basic_info']['rows']} rows × {profile['basic_info']['columns']} columns. "
            f"Quality: {qs}/100 ({profile['quality_score']['rating']}). "
            f"Task: {profile['ml_problem_type']['suggested_type'].replace('_',' ').title()}."
        )
        return rec


# ─── Data Preprocessor ──────────────────────────────────────────────────────
class DataPreprocessor:

    def __init__(self, df: pd.DataFrame, recommendations: Dict):
        self.df  = df.copy()
        self.rec = recommendations
        self.log = []

    def apply_all_preprocessing(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        self._drop_columns()
        self._remove_duplicates()
        self._handle_missing_values()
        self._encode_categoricals()
        self._handle_outliers()
        self._scale_features()
        return (*self._split_data(), self.log)

    def _drop_columns(self):
        cols = [i['column'] for i in self.rec.get('columns_to_drop', [])]
        if cols:
            self.df = self.df.drop(columns=cols, errors='ignore')
            self.log.append(f"Dropped {len(cols)} column(s): {', '.join(cols[:3])}{'...' if len(cols)>3 else ''}")

    def _remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = before - len(self.df)
        if removed:
            self.log.append(f"Removed {removed} duplicate rows ({before} → {len(self.df)})")

    def _handle_missing_values(self):
        filled = 0
        strategy = self.rec.get('missing_value_strategy', {})

        # Always impute ALL columns with missing values
        # Use recommended strategy if available, otherwise auto-detect
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
            method = strategy.get(col)
            if not method:
                # Auto-detect: numeric → median, categorical → mode
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    method = 'median'
                else:
                    method = 'mode'

            if   method == 'mean':
                self.df[col].fillna(self.df[col].mean(), inplace=True); filled += 1
            elif method == 'median':
                self.df[col].fillna(self.df[col].median(), inplace=True); filled += 1
            elif method == 'mode':
                m = self.df[col].mode()
                if len(m): self.df[col].fillna(m[0], inplace=True); filled += 1
            elif method == 'drop_rows':
                before = len(self.df)
                self.df.dropna(subset=[col], inplace=True)
                self.log.append(f"Dropped {before - len(self.df)} rows with missing '{col}'")

        if filled: self.log.append(f"Imputed missing values in {filled} column(s)")

    def _encode_categoricals(self):
        encoded = 0
        for col, enc in self.rec.get('encoding_recommendations', {}).items():
            if col not in self.df.columns: continue
            if enc == 'label_encoding':
                self.df[col] = pd.Categorical(self.df[col]).codes; encoded += 1
            elif enc == 'one_hot_encoding':
                self.df = pd.concat([self.df, pd.get_dummies(self.df[col], prefix=col)], axis=1).drop(columns=[col]); encoded += 1
        if encoded: self.log.append(f"Encoded {encoded} categorical column(s)")

    def _handle_outliers(self):
        if self.rec.get('outlier_handling') != 'iqr': return
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            self.df[col] = self.df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
        if len(num_cols): self.log.append(f"IQR outlier capping on {len(num_cols)} column(s)")

    def _scale_features(self):
        scaling  = self.rec.get('scaling_recommendation', 'standard')
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if not len(num_cols) or scaling == 'none': return
        scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}.get(scaling, StandardScaler())
        self.df[num_cols] = scaler.fit_transform(self.df[num_cols])
        self.log.append(f"{scaling.title()} scaling on {len(num_cols)} column(s)")

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split = self.rec.get('data_split', {'train': 0.7, 'validation': 0.15, 'test': 0.15})
        tv, test   = train_test_split(self.df, test_size=split['test'], random_state=42)
        train, val = train_test_split(tv, test_size=split['validation']/(split['train']+split['validation']), random_state=42)
        self.log.append(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)} rows")
        return train, val, test


# ─── Flask Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    resp = make_response(send_from_directory('static', 'index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file in request'}), 400
        file = request.files['file']
        if not file.filename or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file — use CSV, XLSX or XLS'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        df      = pd.read_csv(filepath) if filename.lower().endswith('.csv') else pd.read_excel(filepath)
        profile = DatasetProfiler(df).analyze()
        profile['filename'] = filename
        return jsonify({'success': True, 'profile': profile})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data    = request.json or {}
        profile = data.get('profile')
        if not profile:
            return jsonify({'error': 'No profile data'}), 400
        rec = PreprocessingEngine().generate_recommendations(profile)
        return jsonify({'success': True, 'recommendations': rec})
    except Exception as e:
        try:
            profile = (request.json or {}).get('profile', {})
            rec = PreprocessingEngine()._rule_based_recommendations(profile)
            rec['_fallback_reason'] = str(e)
            return jsonify({'success': True, 'recommendations': rec})
        except Exception as e2:
            return jsonify({'error': str(e2)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_dataset():
    try:
        data    = request.json or {}
        fname   = data.get('filename')
        rec     = data.get('recommendations')
        fpath   = os.path.join(UPLOAD_FOLDER, fname)
        df      = pd.read_csv(fpath) if fname.lower().endswith('.csv') else pd.read_excel(fpath)
        train, val, test, log = DataPreprocessor(df, rec).apply_all_preprocessing()
        base = fname.rsplit('.', 1)[0]
        train.to_csv(os.path.join(PROCESSED_FOLDER, f"{base}_train.csv"), index=False)
        val.to_csv(  os.path.join(PROCESSED_FOLDER, f"{base}_val.csv"),   index=False)
        test.to_csv( os.path.join(PROCESSED_FOLDER, f"{base}_test.csv"),  index=False)
        return jsonify({'success': True,
                        'train_file': f"{base}_train.csv",
                        'val_file':   f"{base}_val.csv",
                        'test_file':  f"{base}_test.csv",
                        'shapes': {'train': list(train.shape), 'val': list(val.shape), 'test': list(test.shape)},
                        'log': log})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)


@app.route('/api/test-connection', methods=['POST'])
def test_connection():
    try:
        if not _use_groq():
            return jsonify({'success': False,
                            'error': 'No Groq key in app.py — set GROQ_API_KEY = "gsk_..."',
                            'mode': 'rule_based'}), 400
        client = Groq(api_key=GROQ_API_KEY)
        print(f"[test-connection] Using key: {GROQ_API_KEY[:12]}...{GROQ_API_KEY[-4:]} (len={len(GROQ_API_KEY)})")
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Reply with only the word: OK"}],
            max_tokens=5, temperature=0)
        labels = {'mixtral-8x7b-32768': 'Mixtral 8x7B', 'llama3-70b-8192': 'LLaMA 3 70B', 'gemma2-9b-it': 'Gemma 2 9B'}
        return jsonify({'success': True, 'provider': 'groq', 'model': GROQ_MODEL,
                        'model_name': labels.get(GROQ_MODEL, GROQ_MODEL),
                        'reply': resp.choices[0].message.content.strip()})
    except Exception as e:
        err = str(e)
        print(f"[test-connection error] {err}")   # ← shows in terminal
        if '401' in err or 'invalid' in err.lower(): err = 'Invalid API key — check your gsk_... key'
        elif '429' in err or 'rate' in err.lower():  err = 'Rate limit — wait and retry'
        elif '403' in err:                            err = 'Access denied'
        elif 'model' in err.lower():                 err = f'Model not available: {GROQ_MODEL}'
        return jsonify({'success': False, 'error': err}), 400


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mode = f"Groq AI ({GROQ_MODEL})" if _use_groq() else "Rule-based (set GROQ_API_KEY to enable AI)"
    print(f"\n  DataPrep Engine — mode: {mode}")
    print(f"  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)