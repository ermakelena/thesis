import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

def load_lexicon(filepath):
    pos, neg = set(), set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('!'):
                continue
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

def count_words_by_set(text, word_set):
    if not isinstance(text, str):
        return 0
    words = re.findall(r'\b[а-яё]+\b', text.lower())
    return sum(1 for w in words if w in word_set)


def count_sentiment(text, pos_words, neg_words):
    if not isinstance(text, str):
        return 0, 0
    words = re.findall(r'\b[а-яё]+\b', text.lower())
    pos_cnt = sum(1 for w in words if w in pos_words)
    neg_cnt = sum(1 for w in words if w in neg_words)
    return pos_cnt, neg_cnt

def merge_sentences(row: pd.Series) -> str:
    if pd.notna(row.get('sentences')):
        return ' '.join(str(row['sentences']).split('|'))
    return str(row.get('text', ''))


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['sentences'] = out.apply(merge_sentences, axis=1)
    out['sentences'] = out['sentences'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    if 'source' in out.columns:
        out = out.sort_values(['source', 'paragraph']).reset_index(drop=True)
        out['prev_text'] = out.groupby('source')['sentences'].shift(1).fillna('')
    else:
        out = out.sort_values(['paragraph']).reset_index(drop=True)
        out['prev_text'] = ''
    out['text_with_context'] = out['prev_text'].astype(str) + ' [SEP] ' + out['sentences'].astype(str)
    return out


def extract_ling_features(df: pd.DataFrame, pos_words, neg_words) -> pd.DataFrame:
    """Извлечение 12 лингвистических признаков (с антифразисом)"""
    text_series = df['sentences'].fillna('').astype(str)
    prev_text = df['prev_text'].fillna('').astype(str)

    feats = pd.DataFrame(index=df.index)

    pos_cnt = []
    neg_cnt = []
    for text in text_series:
        p, n = count_sentiment(text, pos_words, neg_words)
        pos_cnt.append(p)
        neg_cnt.append(n)

    feats['pos_cnt'] = pos_cnt
    feats['neg_cnt'] = neg_cnt
    feats['sentiment_diff'] = feats['pos_cnt'] - feats['neg_cnt']
    feats['has_both_sentiment'] = ((feats['pos_cnt'] > 0) & (feats['neg_cnt'] > 0)).astype(int)

    feats['low_style_cnt'] = text_series.apply(lambda x: count_words_by_set(x, LOW_STYLE))
    feats['high_style_cnt'] = text_series.apply(lambda x: count_words_by_set(x, HIGH_STYLE))
    feats['has_both_styles'] = ((feats['low_style_cnt'] > 0) & (feats['high_style_cnt'] > 0)).astype(int)

    feats['has_contrast'] = text_series.str.lower().str.contains(r'\b(?:но|однако|зато)\b', na=False).astype(int)

    sarcasm_markers = [
        'как ни странно', 'как ни удивительно', 'неожиданно',
        'кстати', 'между прочим', 'собственно говоря',
        'конечно же', 'ещё бы', 'в самом деле'
    ]
    pattern_sarcasm = r'\b(?:' + '|'.join(sarcasm_markers) + r')\b'
    feats['has_sarcasm_marker'] = text_series.str.lower().str.contains(pattern_sarcasm, na=False).astype(int)

    intro_words = ['разумеется', 'конечно', 'безусловно', 'естественно', 'очевидно', 'несомненно']
    pattern_intro = r'\b(?:' + '|'.join(intro_words) + r')\b'
    feats['num_intro_words'] = text_series.str.lower().str.count(pattern_intro)

    has_positive = feats['pos_cnt'] > 0
    has_negative_context = (feats['neg_cnt'] > 0) | (feats['low_style_cnt'] > 0)
    feats['has_antiphrasis'] = (has_positive & has_negative_context).astype(int)

    feats['prev_len'] = prev_text.str.len()

    return feats.fillna(0)


def metrics_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        'model': model_name,
        'n': len(y_true),
        'positive_rate': float(np.mean(y_true)),
        'accuracy': float(acc),
        'precision_irony': float(precision),
        'recall_irony': float(recall),
        'f1_irony': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }


def main() -> None:
    pos_words, neg_words = load_lexicon('RuSentiLex2017.txt')
    print(f"Позитивных: {len(pos_words)}, Негативных: {len(neg_words)}")
    print(f"Сниженной лексики: {len(LOW_STYLE)}")
    print(f"Высокой лексики: {len(HIGH_STYLE)}")

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_df = normalize_df(train_df)
    test_df = normalize_df(test_df)

    y_train = train_df['marked irony'].astype(int).values
    y_test = test_df['marked irony'].astype(int).values

    X_train_base = extract_ling_features(train_df, pos_words, neg_words)
    X_test_base = extract_ling_features(test_df, pos_words, neg_words)
    base_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    base_model.fit(X_train_base, y_train)
    y_pred_base = base_model.predict(X_test_base)

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(train_df['text_with_context'])
    X_test_tfidf = tfidf.transform(test_df['text_with_context'])

    X_train_ling = extract_ling_features(train_df, pos_words, neg_words)
    X_test_ling = extract_ling_features(test_df, pos_words, neg_words)
    scaler = StandardScaler()
    X_train_ling_scaled = scaler.fit_transform(X_train_ling)
    X_test_ling_scaled = scaler.transform(X_test_ling)

    X_train_combined = hstack([X_train_tfidf, X_train_ling_scaled])
    X_test_combined = hstack([X_test_tfidf, X_test_ling_scaled])

    combined_model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    combined_model.fit(X_train_combined, y_train)
    y_pred_combined = combined_model.predict(X_test_combined)

    rows = [
        metrics_row('baseline_ling_only', y_test, y_pred_base),
        metrics_row('combined_tfidf_context_ling', y_test, y_pred_combined),
    ]
    result_df = pd.DataFrame(rows)
    result_df.to_csv('comparison_on_same_test.csv', index=False)

    # Сохранение предсказаний
    out_pred = test_df[['paragraph', 'source', 'sentences', 'marked irony']].copy()
    out_pred['baseline_pred'] = y_pred_base
    out_pred['combined_pred'] = y_pred_combined
    out_pred.to_csv('comparison_predictions_on_same_test.csv', index=False)

    # Отчёт
    b = rows[0]
    c = rows[1]
    with open('comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write('СРАВНЕНИЕ BASELINE И COMBINED НА ОДНОМ TEST\n')
        f.write('=' * 60 + '\n\n')
        f.write(f"Размер test: {b['n']}\n")
        f.write(f"Доля иронии: {b['positive_rate']:.3f} ({int(round(b['positive_rate']*b['n']))} из {b['n']})\n\n")

        for row in rows:
            f.write(f"[{row['model']}]\n")
            f.write(f"Accuracy:  {row['accuracy']:.3f}\n")
            f.write(f"Precision: {row['precision_irony']:.3f}\n")
            f.write(f"Recall:    {row['recall_irony']:.3f}\n")
            f.write(f"F1:        {row['f1_irony']:.3f}\n")
            f.write(f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}\n\n")

        f.write('ИЗМЕНЕНИЕ (combined - baseline)\n')
        f.write(f"Delta Accuracy:  {c['accuracy'] - b['accuracy']:+.3f}\n")
        f.write(f"Delta Precision: {c['precision_irony'] - b['precision_irony']:+.3f}\n")
        f.write(f"Delta Recall:    {c['recall_irony'] - b['recall_irony']:+.3f}\n")
        f.write(f"Delta F1:        {c['f1_irony'] - b['f1_irony']:+.3f}\n")
        f.write(f"Delta FP:        {c['fp'] - b['fp']:+d}\n")
        f.write(f"Delta FN:        {c['fn'] - b['fn']:+d}\n")

        f.write('\n' + '=' * 60 + '\n')
        f.write('СПИСОК ЛИНГВИСТИЧЕСКИХ ПРИЗНАКОВ (12 шт.)\n')
        f.write('=' * 60 + '\n')
        f.write('1. pos_cnt              - позитивные слова (RuSentiLex)\n')
        f.write('2. neg_cnt              - негативные слова (RuSentiLex)\n')
        f.write('3. sentiment_diff       - pos_cnt - neg_cnt\n')
        f.write('4. has_both_sentiment   - есть и позитивные, и негативные слова\n')
        f.write('5. low_style_cnt        - сниженная/грубая лексика\n')
        f.write('6. high_style_cnt       - высокая/книжная лексика\n')
        f.write('7. has_both_styles      - есть и высокая, и сниженная лексика\n')
        f.write('8. has_contrast         - союзы "но", "однако", "зато"\n')
        f.write('9. has_sarcasm_marker   - маркеры сарказма\n')
        f.write('10. num_intro_words     - вводные слова (разумеется, конечно...)\n')
        f.write('11. has_antiphrasis     - антифразис (позитив + негатив/сниженная лексика)\n')
        f.write('12. prev_len            - длина предыдущего абзаца\n')

    print('Saved: comparison_on_same_test.csv')
    print('Saved: comparison_predictions_on_same_test.csv')
    print('Saved: comparison_report.txt')
    print(result_df.to_string(index=False))


if __name__ == '__main__':
    main()


