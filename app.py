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
            'correlation_analysis': self._analyze_correlation(),
            'feature_importance':  self._basic_feature_importance(),
        }
        profile['quality_score'] = DataQualityScorer.calculate_quality_score(profile)
        return profile

    def _analyze_correlation(self) -> Dict:
        """Find highly correlated feature pairs (>0.9)."""
        try:
            num_df = self.df.select_dtypes(include=[np.number])
            if num_df.shape[1] < 2:
                return {'high_correlation_pairs': [], 'drop_suggestions': []}
            corr = num_df.corr().abs()
            pairs = []
            drop_suggestions = []
            seen = set()
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if val > 0.9:
                        c1, c2 = corr.columns[i], corr.columns[j]
                        pairs.append({'col1': c1, 'col2': c2, 'correlation': round(float(val), 4)})
                        if c2 not in seen:
                            drop_suggestions.append({'column': c2, 'reason': f'Highly correlated with {c1} ({val:.2f})'})
                            seen.add(c2)
            return {'high_correlation_pairs': pairs, 'drop_suggestions': drop_suggestions}
        except Exception:
            return {'high_correlation_pairs': [], 'drop_suggestions': []}

    def _basic_feature_importance(self) -> Dict:
        """Basic feature importance using variance and correlation with last column."""
        try:
            num_df = self.df.select_dtypes(include=[np.number]).dropna()
            if num_df.shape[1] < 2:
                return {'important': [], 'low_importance': []}
            target = num_df.columns[-1]
            correlations = num_df.corr()[target].abs().drop(target).sort_values(ascending=False)
            important     = [{'column': col, 'score': round(float(val), 4)}
                             for col, val in correlations.items() if val >= 0.1]
            low_importance= [{'column': col, 'score': round(float(val), 4)}
                             for col, val in correlations.items() if val < 0.1]
            return {'target_used': target, 'important': important, 'low_importance': low_importance}
        except Exception:
            return {'important': [], 'low_importance': []}

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
            missing_pct = float(self.df[col].isnull().sum() / len(self.df) * 100)
            uc          = int(self.df[col].nunique())
            is_num      = pd.api.types.is_numeric_dtype(self.df[col])

            # Column category
            if self.df[col].dtype == bool or uc == 2:
                col_category = 'Binary'
            elif self.df[col].dtype.name in ['object','category']:
                col_category = 'Text' if uc / max(len(self.df),1) > 0.5 else 'Categorical'
            elif is_num:
                col_category = 'Numeric'
            else:
                col_category = 'Text'

            # Outlier percentage (IQR)
            outlier_pct = 0.0
            skewness    = 0.0
            if is_num and not self.df[col].isnull().all():
                try:
                    Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                    IQR    = Q3 - Q1
                    if IQR > 0:
                        outliers    = ((self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)).sum()
                        outlier_pct = round(float(outliers / len(self.df) * 100), 2)
                    skewness = round(float(self.df[col].skew()), 4)
                except Exception:
                    pass

            # Risk level
            if missing_pct > 30 or outlier_pct > 20:
                risk = 'High'
            elif missing_pct > 10 or outlier_pct > 10:
                risk = 'Medium'
            else:
                risk = 'Low'

            info = {
                'name':               col,
                'dtype':              str(self.df[col].dtype),
                'column_category':    col_category,
                'unique_values':      uc,
                'missing_count':      int(self.df[col].isnull().sum()),
                'missing_percentage': missing_pct,
                'outlier_percentage': outlier_pct,
                'skewness':           skewness,
                'risk_level':         risk,
                'is_constant':        uc == 1,
                'is_identifier':      self._is_identifier_column(col),
                'sample_values':      [str(v) for v in self.df[col].dropna().head(5).tolist()],
            }
            if is_num:
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
        # Only flag as identifier if ALL values are unique (true ID column)
        if self.df[col].nunique() == len(self.df):
            return True
        # Only flag if column NAME clearly indicates an ID
        # Be strict — do not flag demographic/feature columns
        col_lower = col.lower().strip()
        id_patterns = ['_id', 'id_', ' id', 'id ', '_index', 'index_', '_key', 'key_']
        exact_ids   = ['id', 'index', 'key', 'rowid', 'row_id', 'serial', 'uuid']
        if col_lower in exact_ids:
            return True
        if any(col_lower.endswith(p.strip()) or col_lower.startswith(p.strip()) 
               for p in id_patterns):
            return True
        return False

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
        """
        Detect ML problem type by finding the most likely TARGET column.
        Key insight: categorical/boolean columns are almost always FEATURES, not targets.
        The target is typically:
          - A numeric column with very few unique values (classification)
          - OR a numeric column with many unique continuous values (regression)
          - Usually the last column by convention
        """
        n_rows = len(self.df)
        all_candidates = []

        for col in self.df.columns:
            if self._is_identifier_column(col):
                continue
            uc = self.df[col].nunique()
            if uc <= 1:
                continue

            ratio    = uc / n_rows
            is_last  = (col == self.df.columns[-1])
            dtype    = str(self.df[col].dtype)
            is_num   = pd.api.types.is_numeric_dtype(self.df[col])
            is_cat   = self.df[col].dtype.name in ['object', 'category', 'bool']

            # --- Determine this column's ML type ---
            if is_cat:
                if uc == 2:   ml_type = 'binary_classification'
                elif uc <= 20: ml_type = 'multiclass_classification'
                else:          ml_type = 'text'
            elif uc == 2:
                ml_type = 'binary_classification'
            elif uc <= 10:
                ml_type = 'multiclass_classification'
            elif ratio > 0.05:
                ml_type = 'regression'
            else:
                ml_type = 'multiclass_classification'

            # --- Target likelihood score ---
            # Categorical columns are FEATURES in most datasets → penalize heavily
            if is_cat:
                score = 5   # categorical = almost never the target
            elif uc == 2:
                score = 100  # binary numeric = strong target signal
            elif uc <= 5:
                score = 80
            elif uc <= 10:
                score = 60
            elif uc <= 20:
                score = 40
            elif ratio > 0.10:
                score = 70   # high cardinality continuous = good regression target
            else:
                score = 25

            # Last column is conventionally the target
            if is_last and is_num:
                score += 50

            all_candidates.append({
                'column':       col,
                'type':         ml_type,
                'confidence':   'high' if score >= 80 else 'medium' if score >= 40 else 'low',
                'unique_values': int(uc),
                '_score':       score,
            })

        if not all_candidates:
            return {'potential_targets': [], 'suggested_type': 'unknown'}

        best      = max(all_candidates, key=lambda x: x['_score'])
        suggested = best['type']

        targets = [{'column': t['column'], 'type': t['type'],
                    'confidence': t['confidence'], 'unique_values': t['unique_values']}
                   for t in all_candidates]

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
        # Build protected columns list
        protected = [col['name'] for col in profile['column_analysis']
                     if any(kw in col['name'].lower() for kw in
                            ['gender','sex','age','race','ethnicity','education',
                             'city','region','country','state','employment','marital',
                             'diagnosis','target','label','class','result','outcome'])]
        protect_str = ', '.join(protected) if protected else 'none detected'

        # Summarize column info for prompt
        col_lines = []
        for col in profile['column_analysis'][:25]:
            col_lines.append(
                f"  - {col['name']} | {col.get('column_category','?')} | "
                f"missing={col['missing_percentage']:.1f}% | "
                f"outliers={col.get('outlier_percentage',0):.1f}% | "
                f"unique={col['unique_values']} | "
                f"skew={col.get('skewness',0):.2f} | "
                f"risk={col.get('risk_level','?')}"
                + (" [ID]" if col['is_identifier'] else "")
                + (" [CONST]" if col['is_constant'] else "")
            )
        col_summary = chr(10).join(col_lines)

        corr_pairs = profile.get('correlation_analysis', {}).get('high_correlation_pairs', [])
        corr_str   = ', '.join([f"{p['col1']}↔{p['col2']}({p['correlation']})" for p in corr_pairs[:5]]) or 'none'

        prompt = f"""You are an expert data preprocessing system.
Analyze the given dataset and generate a complete preprocessing report with explainable insights.

DATASET SUMMARY
  Rows: {profile['basic_info']['rows']} | Columns: {profile['basic_info']['columns']}
  Missing: {profile['missing_values']['missing_percentage']:.1f}% | Duplicates: {profile['basic_info']['duplicate_rows']}
  Quality Score: {profile['quality_score']['overall']}/100 ({profile['quality_score']['rating']})
  ML Task: {profile['ml_problem_type']['suggested_type']}
  High Correlations: {corr_str}

COLUMN DETAILS
{col_summary}

STRICT RULES:
- NEVER drop: {protect_str}
- Only drop: true row IDs (all values unique), constant columns, or >70% missing
- Do NOT drop gender, age, city, education, employment or any demographic feature
- Be beginner-friendly and explainable

Return ONLY valid JSON (no markdown, no backticks, no extra text):
{{
  "source": "ai_groq",
  "overall_assessment": "<2-3 sentences explaining dataset quality>",
  "columns_to_drop": [{{"column": "<name>", "reason": "<clear reason>"}}],
  "missing_value_strategy": {{"<col>": "mean|median|mode|drop_rows"}},
  "encoding_recommendations": {{"<col>": "one_hot_encoding|label_encoding|target_encoding"}},
  "scaling_recommendation": "standard|minmax|robust|none",
  "outlier_handling": "iqr|zscore|none",
  "data_split": {{"train": 0.7, "validation": 0.15, "test": 0.15}},
  "preprocessing_steps": ["<step1>", "<step2>"],
  "suggestions": [{{"text": "<insight>", "severity": "high|medium|low", "impact": "<effect on model>"}}],
  "feature_importance_notes": [{{"column": "<name>", "importance": "high|medium|low", "reason": "<why>"}}],
  "data_issues": [{{"issue": "<name>", "severity": "high|medium|low", "affected_columns": ["<col>"], "impact": "<ml impact>"}}],
  "before_after_summary": [{{"metric": "<name>", "before": "<value>", "after": "<expected value>"}}]
}}"""

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

            # Safety: remove protected columns from drop list
            protected_kw = ['gender','sex','age','race','ethnicity','education',
                            'city','region','country','state','employment','marital',
                            'diagnosis','target','label','class','result','outcome']
            rec['columns_to_drop'] = [
                d for d in rec.get('columns_to_drop', [])
                if not any(kw in d['column'].lower() for kw in protected_kw)
            ]
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

        # Important feature columns that should NEVER be dropped
        protected_keywords = ['gender', 'sex', 'age', 'race', 'ethnicity', 'education',
                              'city', 'region', 'country', 'state', 'employment', 'marital']

        for ci in profile['column_analysis']:
            col_lower = ci['name'].lower()
            is_protected = any(kw in col_lower for kw in protected_keywords)

            if ci['is_constant']:
                rec['columns_to_drop'].append({'column': ci['name'], 'reason': 'Constant — carries no information'})
            elif ci['is_identifier'] and not is_protected:
                rec['columns_to_drop'].append({'column': ci['name'], 'reason': 'Identifier — not a feature'})
            elif ci['missing_percentage'] > 70 and not is_protected:
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
        # Step 0: Convert ALL boolean columns to 0/1 integers immediately
        bool_cols = [c for c in self.df.columns if self.df[c].dtype == bool
                     or str(self.df[c].dtype) == 'bool'
                     or (self.df[c].dropna().isin([True, False]).all() and self.df[c].dtype == object)]
        for col in bool_cols:
            self.df[col] = self.df[col].map({True: 1, False: 0, 'True': 1, 'False': 0,
                                              'true': 1, 'false': 0, 'TRUE': 1, 'FALSE': 0})
            self.df[col] = self.df[col].astype(float)
        if bool_cols:
            self.log.append(f"Converted {len(bool_cols)} boolean column(s) to 0/1")

        self._drop_columns()
        self._remove_duplicates()
        self._handle_missing_values()
        self._encode_categoricals()
        self._handle_outliers()
        # Save pre-scaled snapshot for preview verification
        self.df_before_scaling = self.df.copy()
        self._scale_features()
        return (*self._split_data(), self.log)

    def _drop_columns(self):
        cols = [i['column'] for i in self.rec.get('columns_to_drop', [])]
        if cols:
            self.df = self.df.drop(columns=cols, errors='ignore')
            self.log.append(f"Dropped {len(cols)} column(s): {', '.join(cols[:3])}{'...' if len(cols)>3 else ''}")

    def _remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        if removed:
            self.log.append(f"Removed {removed} duplicate rows ({before} → {len(self.df)})")

    def _handle_missing_values(self):
        filled = 0
        strategy = self.rec.get('missing_value_strategy', {})

        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue

            # Skip boolean columns
            if self.df[col].dtype == bool:
                continue

            method = strategy.get(col)
            if not method:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    method = 'median'
                else:
                    method = 'mode'

            if method == 'mean':
                fill_val = self.df[col].mean()
                if pd.isna(fill_val): fill_val = 0
                self.df[col] = self.df[col].fillna(round(fill_val, 6))
                filled += 1
            elif method == 'median':
                fill_val = self.df[col].median()
                if pd.isna(fill_val): fill_val = 0
                self.df[col] = self.df[col].fillna(round(fill_val, 6))
                filled += 1
            elif method == 'mode':
                m = self.df[col].mode()
                if len(m):
                    self.df[col] = self.df[col].fillna(m[0])
                    filled += 1
            elif method == 'drop_rows':
                before = len(self.df)
                self.df = self.df.dropna(subset=[col])
                self.log.append(f"Dropped {before - len(self.df)} rows with missing '{col}'")

        if filled:
            self.log.append(f"Imputed missing values in {filled} column(s)")

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
        try:
            all_num = list(self.df.select_dtypes(include=[np.number]).columns)
            num_cols = [col for col in all_num
                        if not (self.df[col].dropna().isin([0, 1]).all() and self.df[col].nunique() <= 2)]
            capped = 0
            for col in num_cols:
                try:
                    Q1  = self.df[col].quantile(0.25)
                    Q3  = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR == 0: continue   # skip constant columns
                    self.df[col] = self.df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                    capped += 1
                except Exception:
                    continue
            if capped:
                self.log.append(f"IQR outlier capping on {capped} continuous column(s)")
        except Exception as e:
            self.log.append(f"Outlier handling skipped: {e}")

    def _scale_features(self):
        scaling  = self.rec.get('scaling_recommendation', 'standard')
        try:
            # Convert any remaining bool columns to int before scaling
            for col in self.df.columns:
                if self.df[col].dtype == bool:
                    self.df[col] = self.df[col].astype(int)
                elif self.df[col].dtype == object:
                    sample = self.df[col].dropna().astype(str).str.lower().unique()
                    if set(sample).issubset({'true', 'false', '1', '0'}):
                        self.df[col] = self.df[col].astype(str).str.lower().map(
                            {'true': 1, 'false': 0, '1': 1, '0': 0}
                        )
            num_cols = list(self.df.select_dtypes(include=[np.number]).columns)
            if not num_cols or scaling == 'none': return
            scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}.get(scaling, StandardScaler())
            scaled = scaler.fit_transform(self.df[num_cols])
            self.df[num_cols] = np.round(scaled, 6)
            self.log.append(f"{scaling.title()} scaling on {len(num_cols)} column(s)")
        except Exception as e:
            self.log.append(f"Scaling skipped: {e}")

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        split = self.rec.get('data_split', {'train': 0.7, 'validation': 0.15, 'test': 0.15})
        tv, test   = train_test_split(self.df, test_size=split['test'], random_state=42)
        train, val = train_test_split(tv, test_size=split['validation']/(split['train']+split['validation']), random_state=42)
        self.log.append(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)} rows")
        return train, val, test

    def _clean_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup before saving: convert all bool/object bool to 0/1."""
        df = df.copy()
        for col in df.columns:
            # Convert boolean dtype
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
            # Convert string 'True'/'False' or 'true'/'false'
            elif df[col].dtype == object:
                sample = df[col].dropna().astype(str).str.lower().unique()
                if set(sample).issubset({'true', 'false', '1', '0'}):
                    df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0})
        return df


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
        dp = DataPreprocessor(df, rec)
        train, val, test, log = dp.apply_all_preprocessing()
        base = fname.rsplit('.', 1)[0]

        # Save fully processed (scaled + encoded) splits as main download
        dp._clean_for_export(train).to_csv(os.path.join(PROCESSED_FOLDER, f"{base}_train.csv"), index=False)
        dp._clean_for_export(val).to_csv(  os.path.join(PROCESSED_FOLDER, f"{base}_val.csv"),   index=False)
        dp._clean_for_export(test).to_csv( os.path.join(PROCESSED_FOLDER, f"{base}_test.csv"),  index=False)

        # Save pre-scaled snapshot for preview only
        if hasattr(dp, 'df_before_scaling') and dp.df_before_scaling is not None:
            pre = dp._clean_for_export(dp.df_before_scaling)
            split_cfg = dp.rec.get('data_split', {'train':0.7,'validation':0.15,'test':0.15})
            tv_pre, test_pre   = train_test_split(pre, test_size=split_cfg['test'], random_state=42)
            val_ratio          = split_cfg['validation']/(split_cfg['train']+split_cfg['validation'])
            train_pre, val_pre = train_test_split(tv_pre, test_size=val_ratio, random_state=42)
            train_pre.to_csv(os.path.join(PROCESSED_FOLDER, f"{base}_preview.csv"), index=False)
        return jsonify({'success': True,
                        'train_file': f"{base}_train.csv",
                        'val_file':   f"{base}_val.csv",
                        'test_file':  f"{base}_test.csv",
                        'shapes': {'train': list(train.shape), 'val': list(val.shape), 'test': list(test.shape)},
                        'log': log})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview/<filename>')
def preview_file(filename):
    """Return first 20 rows of a processed file for UI preview."""
    try:
        # Use pre-scaled preview file per split
        preview_filename = filename.replace('_train.csv','_train_preview.csv').replace('_val.csv','_val_preview.csv').replace('_test.csv','_test_preview.csv')
        preview_path = os.path.join(PROCESSED_FOLDER, preview_filename)
        filepath = preview_path if os.path.exists(preview_path) else os.path.join(PROCESSED_FOLDER, filename)
        df = pd.read_csv(filepath)

        # Convert any remaining bool-string columns to 0/1
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).str.lower().unique()
                if set(sample).issubset({'true', 'false', '1', '0'}):
                    df[col] = df[col].astype(str).str.lower().map(
                        {'true': 1, 'false': 0, '1': 1, '0': 0}
                    )
            elif df[col].dtype == bool:
                df[col] = df[col].astype(int)

        preview = df.head(20).where(pd.notnull(df.head(20)), None)
        return jsonify({
            'success': True,
            'columns': list(df.columns),
            'rows':    preview.values.tolist(),
            'shape':   list(df.shape),
            'dtypes':  {col: str(df[col].dtype) for col in df.columns},
            'missing': int(df.isnull().sum().sum()),
        })
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