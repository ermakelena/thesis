import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import re
import warnings

warnings.filterwarnings('ignore')

from experiment_outputs import add_binary_outcomes, save_predictions_excel

def load_lexicon(filepath):
    pos, neg = set(), set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'): continue
            parts = line.strip().split(', ')
            if len(parts) >= 4:
                word = parts[0].lower()
                sent = parts[3]
                if sent == 'positive':
                    pos.add(word)
                elif sent == 'negative':
                    neg.add(word)
    return pos, neg

LOW_STYLE = {
    'сука', 'фраер', 'бля', 'дерьмо', 'хрен',  'сволочь', 'мразь', 'урод',
     'подонок',  'ублюдок', 'мудозвон', 'говно', 'жрать', 'хавать',
    'нажраться', 'надраться', 'спьяну',  'алкаш', 'пьянь',
    'бухать', 'курва', 'потаскуха', 'тварь', 'гнида', 'падла', 'мудак',
    'козел',   'раздолбай', 'жлоб',
    'жулик','шпана', 'банда', 'ворюга',
     'наехать', 'кинуть', 'развести', 'подстава', 'беспредел', 'отморозок',
    'мусор', 'ментовка', 'обезьянник', 'харя', 'рыло', 'морда', 'рожа',
     'угарать', 'барыга', 'толкать', 'гавно', 'гандон', 'говнюк'
}

HIGH_STYLE = {
    'взыграть', 'восскорбеть', 'возроптать', 'убояться', 'истовый', 'токмо', 'ибо',
    'посему', 'отнюдь', 'невзирая', 'денно', 'нощно', 'всечасно', 'неукоснительно',
    'восчувствовать', 'возжаждать', 'воспылать', 'презреть', 'устыдиться', 'ужаснуться',
    'устрашиться', 'озариться', 'просветлеть', 'возликовать', 'воссиять', 'узреть',
    'внемлить', 'изречь', 'гласить', 'молвить', 'дерзать', 'вкушать', 'испить',
    'отведать', 'насытиться', 'уповать', 'чаять', 'зреть', 'взирать', 'очами',
    'устами', 'челом', 'виждь', 'ведай', 'твори', 'мнится', 'рекомый', 'нарицаемый',
    'вельми', 'зело', 'изрядно', 'донельзя', 'невозбранно', 'всуе', 'вотще', 'тщетно',
    'бесследно', 'безвозвратно', 'безвестно', 'безгласно', 'безмолвно', 'безропотно',
    'безответно', 'безотчетно', 'бездумно', 'безрассудно', 'безоглядно', 'бестрепетно',
    'безбоязненно', 'бесстрашно', 'неукротимо', 'неудержимо', 'неотвратимо', 'непреложно',
    'незыблемо', 'неколебимо', 'непоколебимо', 'несокрушимо', 'неразрывно', 'неразлучно',
    'нерасторжимо', 'нераздельно', 'преобразиться', 'обновиться', 'воскреснуть',
    'возродиться', 'воспрянуть', 'воодушевиться', 'окрылиться', 'преисполниться',
    'проникнуться', 'исполниться', 'наполниться', 'претвориться', 'воплотиться',
    'облагодетельствовать', 'осчастливить', 'одарить', 'пожаловать', 'соблаговолить',
    'удостоить', 'сподобить', 'увенчать', 'короновать', 'венчать', 'благой', 'благостный',
    'благодатный', 'благотворный', 'благоуханный', 'благочестивый', 'богобоязненный',
    'божественный', 'возвышенный', 'вседержитель', 'всесильный', 'всемогущий', 'святый',
    'святой', 'священный', 'непорочный', 'непогрешимый', 'непререкаемый', 'незыблемый',
    'неколебимый', 'непоколебимый', 'непреложный', 'неопровержимый', 'неоспоримый',
    'несомненный', 'бесспорный', 'безусловный', 'абсолютный', 'совершенный'
}

pos_words, neg_words = load_lexicon('RuSentiLex2017.txt')
print(f"Позитивных: {len(pos_words)}, Негативных: {len(neg_words)}")
print(f"Сниженной лексики: {len(LOW_STYLE)}")
print(f"Высокой лексики: {len(HIGH_STYLE)}")

df = pd.read_excel('абзацы для аннотации.xlsx', sheet_name='Sheet1')
print(f"Исходный размер: {df.shape}")
print(f"Ирония: {df['marked irony'].sum()} ({100 * df['marked irony'].mean():.1f}%)")


def merge_sentences(row):
    if pd.notna(row['sentences']):
        return ' '.join(row['sentences'].split('|'))
    else:
        return row['text']


df['sentences'] = df.apply(merge_sentences, axis=1)
df['sentences'] = df['sentences'].str.replace(r'\s+', ' ', regex=True).str.strip()

df = df.sort_values(['source', 'paragraph'])
df['prev_text'] = df.groupby('source')['sentences'].shift(1)
df['prev_text'] = df['prev_text'].fillna('')
df['text_with_context'] = df['prev_text'] + " [SEP] " + df['sentences']


def count_words_by_set(text, word_set):
    if not isinstance(text, str):
        return 0
    words = re.findall(r'\b[а-яё]+\b', text.lower())
    return sum(1 for w in words if w in word_set)


def count_sentiment(text):
    if not isinstance(text, str):
        return 0, 0
    words = re.findall(r'\b[а-яё]+\b', text.lower())
    return sum(1 for w in words if w in pos_words), sum(1 for w in words if w in neg_words)

def extract_linguistic_features(df):
    df_result = df.copy()
    text_series = df_result['sentences'].fillna('')
    prev_series = df_result['prev_text'].fillna('')

    df_result['pos_cnt'], df_result['neg_cnt'] = zip(*text_series.apply(count_sentiment))
    df_result['sentiment_diff'] = df_result['pos_cnt'] - df_result['neg_cnt']
    df_result['has_both_sentiment'] = ((df_result['pos_cnt'] > 0) & (df_result['neg_cnt'] > 0)).astype(int)

    df_result['low_style_cnt'] = text_series.apply(lambda x: count_words_by_set(x, LOW_STYLE))
    df_result['high_style_cnt'] = text_series.apply(lambda x: count_words_by_set(x, HIGH_STYLE))
    df_result['has_both_styles'] = ((df_result['low_style_cnt'] > 0) & (df_result['high_style_cnt'] > 0)).astype(int)

    df_result['has_contrast'] = text_series.str.contains(r'(?:но|однако|зато)', case=False, na=False).astype(int)

    sarcasm_markers = r'(?:как ни странно|как ни удивительно|неожиданно|кстати|конечно же|разумеется|ещё бы|в самом деле)'
    df_result['has_sarcasm_marker'] = text_series.str.contains(sarcasm_markers, case=False, na=False).astype(int)

    intro_words = r'\b(?:разумеется|конечно|безусловно|естественно|очевидно|несомненно)\b'
    df_result['num_intro_words'] = text_series.str.lower().str.count(intro_words)

    has_positive = df_result['pos_cnt'] > 0
    has_negative_context = (df_result['neg_cnt'] > 0) | (df_result['low_style_cnt'] > 0)
    df_result['has_antiphrasis'] = (has_positive & has_negative_context).astype(int)

    df_result['prev_len'] = prev_series.str.len()

    ling_columns = [
        'pos_cnt', 'neg_cnt', 'sentiment_diff', 'has_both_sentiment',
        'low_style_cnt', 'high_style_cnt', 'has_both_styles',
        'has_contrast', 'has_sarcasm_marker', 'num_intro_words',
        'has_antiphrasis',
        'prev_len'
    ]

    for col in ling_columns:
        df_result[col] = df_result[col].fillna(0)

    return df_result, ling_columns

df, ling_columns = extract_linguistic_features(df)
print(f"Извлечено признаков: {len(ling_columns)}")
print(f"Признаки: {ling_columns}")

print("СТРАТИФИЦИРОВАННАЯ КРОСС-ВАЛИДАЦИЯ (5-fold)")

X_text = df['text_with_context'].values
X_linguistic = df[ling_columns].values
y = df['marked irony'].values

tfidf_params = {
    'ngram_range': (1, 2),
    'max_features': 3000,
    'min_df': 2,
    'max_df': 0.95,
}

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=1.0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        max_depth=10
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {name: {'f1': [], 'precision': [], 'recall': []} for name in models.keys()}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y), 1):
    print(f"\nFold {fold}")

    X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
    X_ling_train, X_ling_val = X_linguistic[train_idx], X_linguistic[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    tfidf = TfidfVectorizer(**tfidf_params)
    X_tfidf_train = tfidf.fit_transform(X_text_train)
    X_tfidf_val = tfidf.transform(X_text_val)

    scaler = StandardScaler()
    X_ling_train_scaled = scaler.fit_transform(X_ling_train)
    X_ling_val_scaled = scaler.transform(X_ling_val)

    X_train_combined = hstack([X_tfidf_train, csr_matrix(X_ling_train_scaled)])
    X_val_combined = hstack([X_tfidf_val, csr_matrix(X_ling_val_scaled)])

    print(f"  TF-IDF размерность: {X_tfidf_train.shape[1]}")
    print(f"  Лингвистических признаков: {X_ling_train.shape[1]}")
    print(f"  Итого признаков: {X_train_combined.shape[1]}")
    print(f"  Train: {len(y_train)} (ирония: {y_train.sum()})")
    print(f"  Val: {len(y_val)} (ирония: {y_val.sum()})")

    for name, model in models.items():
        model.fit(X_train_combined, y_train)
        y_pred = model.predict(X_val_combined)

        f1 = f1_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)

        cv_results[name]['f1'].append(f1)
        cv_results[name]['precision'].append(prec)
        cv_results[name]['recall'].append(rec)

        print(f"    {name:25} | F1={f1:.3f} | P={prec:.3f} | R={rec:.3f}")

print("ИТОГИ КРОСС-ВАЛИДАЦИИ (среднее +- std)")

for name in models.keys():
    mean_f1 = np.mean(cv_results[name]['f1'])
    std_f1 = np.std(cv_results[name]['f1'])
    mean_prec = np.mean(cv_results[name]['precision'])
    mean_rec = np.mean(cv_results[name]['recall'])

    print(f"{name:25} | F1 = {mean_f1:.3f} ± {std_f1:.3f} | P={mean_prec:.3f} | R={mean_rec:.3f}")

tfidf_final = TfidfVectorizer(**tfidf_params)
X_tfidf_full = tfidf_final.fit_transform(X_text)

scaler_final = StandardScaler()
X_ling_full_scaled = scaler_final.fit_transform(X_linguistic)

X_full_combined = hstack([X_tfidf_full, csr_matrix(X_ling_full_scaled)])

best_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    C=1.0
)
best_model.fit(X_full_combined, y)

print(f"Финальная модель: Logistic Regression")
print(f"Размерность признаков: {X_full_combined.shape[1]}")
print(f"  - TF-IDF: {X_tfidf_full.shape[1]}")
print(f"  - Лингвистические: {X_ling_full_scaled.shape[1]}")
print(f"Обучающих примеров: {len(y)}")



df['combined_pred'] = best_model.predict(X_full_combined)
df['combined_proba_irony'] = best_model.predict_proba(X_full_combined)[:, 1]

df.to_csv('combined_predictions.csv', index=False)
print("\nСохранено: combined_predictions.csv")


print(f"Всего примеров: {len(df)}")
print(f"Реальная ирония: {y.sum()} ({100 * y.mean():.1f}%)")
print(f"Предсказанная ирония: {df['combined_pred'].sum()} ({100 * df['combined_pred'].mean():.1f}%)")
print(
    f"Совпало предсказаний: {(df['marked irony'] == df['combined_pred']).sum()} ({100 * (df['marked irony'] == df['combined_pred']).mean():.1f}%)")

cm = confusion_matrix(df['marked irony'], df['combined_pred'])

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NO IRONY', 'IRONY'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix: TF-IDF + Linguistic Features', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_combined.png', dpi=150)
plt.close()

print("\nМатрица ошибок:")
print(f"  True Negatives (NO → NO):     {cm[0, 0]}")
print(f"  False Positives (NO → IRONY): {cm[0, 1]}")
print(f"  False Negatives (IRONY → NO): {cm[1, 0]}")
print(f"  True Positives (IRONY → IRONY): {cm[1, 1]}")



f1_final = f1_score(y, df['combined_pred'], zero_division=0)

results_comparison = {
    'TF-IDF + 12 лингвистических признаков': f1_final
}

for method, f1 in results_comparison.items():
    if f1 is not None:
        print(f"{method:40} | F1 = {f1:.3f}")


fn = df[(df['marked irony'] == 1) & (df['combined_pred'] == 0)]
fp = df[(df['marked irony'] == 0) & (df['combined_pred'] == 1)]
tp = df[(df['marked irony'] == 1) & (df['combined_pred'] == 1)]



