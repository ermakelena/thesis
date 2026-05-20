import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

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
df['text'] = df.apply(lambda r: ' '.join(r['sentences'].split('|')) if pd.notna(r['sentences']) else r['text'], axis=1)
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

df = df.sort_values(['paragraph'])
df['prev_text'] = df.groupby('source')['text'].shift(1).fillna('')

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

df['pos_cnt'], df['neg_cnt'] = zip(*df['text'].apply(count_sentiment))
df['sentiment_diff'] = df['pos_cnt'] - df['neg_cnt']
df['has_both_sentiment'] = ((df['pos_cnt'] > 0) & (df['neg_cnt'] > 0)).astype(int)

df['low_style_cnt'] = df['text'].apply(lambda x: count_words_by_set(x, LOW_STYLE))
df['high_style_cnt'] = df['text'].apply(lambda x: count_words_by_set(x, HIGH_STYLE))
df['has_both_styles'] = ((df['low_style_cnt'] > 0) & (df['high_style_cnt'] > 0)).astype(int)

df['has_contrast'] = df['text'].str.contains(r'(?:но|однако|зато)', case=False, na=False).astype(int)

sarcasm_markers = r'(?:как ни странно|как ни удивительно|неожиданно|кстати|конечно же|разумеется|ещё бы|в самом деле)'
df['has_sarcasm_marker'] = df['text'].str.contains(sarcasm_markers, case=False, na=False).astype(int)

intro_words = r'\b(?:разумеется|конечно|безусловно|естественно|очевидно|несомненно)\b'
df['num_intro_words'] = df['text'].str.lower().str.count(intro_words)

has_positive = df['pos_cnt'] > 0
has_negative_context = (df['neg_cnt'] > 0) | (df['low_style_cnt'] > 0)
df['has_antiphrasis'] = (has_positive & has_negative_context).astype(int)

df['prev_len'] = df['prev_text'].str.len()

feature_cols = [
    'pos_cnt', 'neg_cnt', 'sentiment_diff', 'has_both_sentiment',
    'low_style_cnt', 'high_style_cnt', 'has_both_styles',
    'has_contrast', 'has_sarcasm_marker', 'num_intro_words',
    'has_antiphrasis']

X = df[feature_cols].fillna(0)
y = df['marked irony']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

results = {}

for name, model in models.items():
    X_use = X_train_scaled if name == 'SVM' else X_train
    X_val_use = X_val_scaled if name == 'SVM' else X_val

    model.fit(X_use, y_train)
    y_pred = model.predict(X_val_use)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    results[name] = (model, f1)
    print(f"{name}: F1 = {f1:.3f}, Precision = {precision:.3f}, Recall = {recall:.3f}")

best_name, (best_model, best_f1) = max(results.items(), key=lambda x: x[1][1])
print(f"\nЛучшая модель на валидации: {best_name} (F1 = {best_f1:.3f})")

X_test_use = X_test_scaled if best_name == 'SVM' else X_test
y_pred = best_model.predict(X_test_use)

print(classification_report(y_test, y_pred, target_names=['NO IRONY', 'IRONY'], digits=3))
print(f"F1: {f1_score(y_test, y_pred):.3f}")

print("\nМатрица ошибок на тесте:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NO IRONY', 'IRONY'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix on Test Set')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Матрица ошибок сохранена как 'confusion_matrix.png'")

df_test = df.iloc[X_test.index].copy()
df_test['pred'] = y_pred
df_test[['paragraph', 'source', 'text', 'marked irony', 'pred']].to_csv('predictions_clean.csv', index=False)
print("\nСохранено: predictions_clean.csv")