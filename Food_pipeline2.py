import pandas as pd
import numpy as np
import re
import unicodedata
import warnings
import os

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ── 1. ĐỊNH NGHĨA ĐƯỜNG DẪN ──
OUT = r"D:\EDA\Category_pipelines\food"
full_path = os.path.join(OUT, "foodfood_full_labeled.csv")

if not os.path.exists(full_path):
    raise FileNotFoundError(f"Không tìm thấy file tại: {full_path}. Hãy kiểm tra lại!")

df = pd.read_csv(full_path)

# ── 2. ĐỊNH NGHĨA RULES (EXCLUSION & INCLUSION) ──
EXCLUSION_PREFIXES = [
    r'\b(cơm|com)\s*(chiên|chien|rang|tấm|tam|trộn|tron)\b',
    r'\b(mì|mi)\s*(xào|xao|goreng|gói|tôm|tom|sườn|suon|gà|ga)\b',
    r'\b(phở|pho)\s*(bò|bo|gà|ga|heo)\b',
    r'\b(bún|bun)\s*(bò|bo|gà|ga|heo|chả|cha)\b',
    r'\b(cháo|chao)\s*(gà|ga|heo|bò|bo)\b',
    r'\b(bánh|banh)\s*(mì|mi|cuốn|cuon)\b',
    r'\b(lẩu|lau)\s*(hải sản|hai san|bò|bo|gà|ga)\b',
    r'\b(combo|set)\s*(cơm|com|bún|bun|phở|pho|gà|ga|bò|bo)\b',
    r'\b(hủ tiếu|hu tieu)\b',
    r'\b(ramen|udon|soba)\b',
    r'\bkim\s*chi\b',
    r'\bsnack\b.*\b(gà|ga|heo|bò|bo|tôm|tom)\b',
    r'\b(bánh quy|banh quy|cracker)\b.*\b(gà|ga|heo|bò|bo)\b',
    r'\b(vifon|acecook|omachi|kokomi|hảo hảo|hao hao)\b',
    r'\b(sốt|sot)\s*(bbq|tiêu|tieu|chanh|gừng|gung)\b',
    r'\btôm\s*xào\s*chua\s*ngọt\b', r'\btom\s*xao\s*chua\s*ngot\b',
    r'\bhuong\s*vi\s*(suon|sườn|tom|tôm|ga|gà|bo|bò)\b',
    r'\bvị\s*(sườn|tôm|gà|bò)\b',
]

MEAT_RULES_V3 = [
    # Thịt heo — phần thịt thô
    r'\bnạc\s*(heo|vai|đùi|rọi|dăm)\b', r'\bnac\s*(heo|vai|dui|roi|dam)\b',
    r'\bba\s*rọi\b', r'\bba\s*roi\b',
    r'\bba\s*chỉ\s*heo\b', r'\bba\s*chi\s*heo\b',
    r'\bcốt\s*lết\b', r'\bcot\s*let\b',
    r'\bchân\s*giò\s*heo\b', r'\bchan\s*gio\s*heo\b',
    r'\bgiò\s*heo\b', r'\bgio\s*heo\b',
    r'\bsườn\s*(non|nướng|cắt|ram|khúc)\b', r'\bsuon\s*(non|nuong|cat|ram|khuc)\b',
    r'\bđùi\s*heo\b', r'\bdui\s*heo\b',
    r'\bnạc\s*rọi\b', r'\bnac\s*roi\b',
    r'\bheo\s*(xay|quay|sữa|rừng|iberico)\b', r'\bheo\s*(xay|quay|sua|rung)\b',
    r'\bmỡ\s*heo\b', r'\bmo\s*heo\b',
    r'\bxương\s*cổ\s*heo\b', r'\bxuong\s*co\s*heo\b',
    r'\bbắp\s*giò\b', r'\bbap\s*gio\b',
    r'\bheo\s*sữa\s*quay\b', r'\bheo\s*rừng\b',
    # Thịt bò tươi
    r'\bbò\s*(mỹ|wagyu|nhúng|tươi|cuộn)\b', r'\bbo\s*(my|wagyu|nhung|tuoi|cuon)\b',
    r'\blưỡi\s*bò\b', r'\bluoi\s*bo\b',
    r'\btổ\s*ong\s*bò\b', r'\bto\s*ong\s*bo\b',
    r'\bthăn\s*(ngoài|trong)?\s*bò\b', r'\bthan\s*(ngoai|trong)?\s*bo\b',
    r'\bnabe\s*bò\b', r'\bnabe\s*bo\b',
    # Thịt gà tươi / miếng
    r'\bcánh\s*gà\b', r'\bcanh\s*ga\b',
    r'\bức\s*gà\b', r'\buc\s*ga\b',
    r'\bđùi\s*gà\b', r'\bdui\s*ga\b',
    r'\bmá\s*đùi\s*gà\b', r'\bma\s*dui\s*ga\b',
    r'\bgà\s*(ta|slim)\b', r'\bga\s*(ta|slim)\b',
    r'\b3f\s*(ức|cánh|đùi|uc|canh|dui)\b',
    # Cá tươi / fillet / hải sản tươi
    r'\bcá\s*(nục|hồi|thu|basa|trê|ngừ|tuyết|lóc|tra|bống|chim|mú|rô|bớp|điêu|trích)\b',
    r'\bca\s*(nuc|hoi|thu|basa|tre|ngu|tuyet|loc|tra|bong|chim|mu|ro|bop|dieu|trich)\b',
    r'\bca\s*(chinh|song|hu|ngan|dia|bop|bau)\b',
    r'\bsalmon\b', r'\bca\s*hoi\b',
    r'\bphi\s*lê\b', r'\bphi\s*le\b',
    r'\bfillet\s*(cá|ca|heo|bò|bo|gà|ga)\b',
    r'\bcrispy\s*salmon\b', r'\bgrilled\s*salmon\b',
    r'\bsalmon\s*(roll|head|skin|teriyaki|maki)\b',
    r'\bcá\s*hộp\b', r'\bca\s*hop\b',
    r'\bcá\s*viên\b', r'\bca\s*vien\b',
    r'\bchả\s*cá\b', r'\bcha\s*ca\b',
    r'\bsteamed\s*fish\b', r'\bfish\b(?!\s*(sauce|finger|cake|oil|ball))',
    r'\btuna\b', r'\btemaki\b', r'\bgunkan\b',
    # Hải sản
    r'\bhải\s*sản\s*(đặc biệt|hấp|nướng)\b', r'\bhai\s*san\s*(dac biet|hap|nuong)\b',
    r'\btôm\s*(sú|hùm|tích|tươi)\b', r'\btom\s*(su|hum|tich|tuoi)\b',
    r'\bmực\s*(tươi|nguyên con)\b', r'\bmuc\s*(tuoi|nguyen con)\b',
    r'\bcua\s*(biển|tươi)\b', r'\bcua\s*(bien|tuoi)\b',
    r'\bghẹ\b', r'\bghe\b', r'\bnghêu\b', r'\bngheu\b',
    r'\bbạch\s*tuộc\b', r'\bbạch\s*tuoc\b',
    r'\blươn\b', r'\bbào\s*ngư\b', r'\bbao\s*ngu\b',
    # Trứng
    r'\btrứng\s*(gà|vịt|cút)\b(?!\s*(chiên|xào|bác))',
    r'\btrung\s*(ga|vit|cut)\b(?!\s*(chien|xao|bac))',
    r'\bboiled\s*egg\b', r'\bfreid\s*egg\b',
    # Brand + thịt đi kèm
    r'\bvs990\b', r'\bves990\b', r'\bnp990\b',
    r'\bmeat\s*(deli|master)\b', r'\bjapfa\b',
    r'\b(heo|gà|ga|bò|bo)\s*(cp|c\.p)\b',
    r'\b(cp|c\.p)\s*(heo|gà|ga|bò|bo)\b',
    # Tiếng Anh
    r'\bpork\s*(neck|belly|chop|ribs?|loin)\b',
    r'\bbeef\s*(slice|roll|wagyu|brisket)\b',
    r'\bchicken\s*(breast|thigh|wing|leg)\b(?!\s*(fried|chiên|nướng gói))',
    r'\bshrimp\s*(scampi|tempura)\b',
    # Món đặc thù
    r'\btonkatsu\b', r'\bgỏi\s*cá\b', r'\bgoi\s*ca\b',
    r'\bcanh\s*giò\s*heo\b', r'\bcanh\s*gio\s*heo\b',
    r'\brtc\s*(canh|gio)\b', r'\bnew\s*cut.*\b(heo|bò|bo|gà|ga)\b',
]

# ── 3. HÀM XỬ LÝ TEXT & ÁP DỤNG RULES ──
def is_exclusion(text):
    tl = str(text).lower() if pd.notnull(text) else ''
    for pat in EXCLUSION_PREFIXES:
        try:
            if re.search(pat, tl, re.IGNORECASE): return True
        except: pass
    return False

def match_meat_v3(text):
    if is_exclusion(text): return False
    tl = str(text).lower() if pd.notnull(text) else ''
    for pat in MEAT_RULES_V3:
        try:
            if re.search(pat, tl, re.IGNORECASE): return True
        except: pass
    return False

UNIT_RE = re.compile(r'\b\d+[\.,]?\d*\s*(?:kg|g|gram|ml|l|gói|hộp|túi|cái|miếng|lát|trái|quả|củ|bó|lon|chai|thùng|bịch|set|pcs?|phần|suất|vé)\b', re.IGNORECASE)

def preprocess(text):
    if not text or pd.isnull(text): return ''
    t = unicodedata.normalize('NFC', str(text)).lower()
    t = UNIT_RE.sub(' ', t)
    t = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF\-&+/]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

# ── 4. XỬ LÝ DỮ LIỆU & KIỂM TRA ĐỘ CHÍNH XÁC RULES ──
print("Đang tiền xử lý văn bản...")
df['clean'] = df['item_name'].apply(preprocess)

# Lưu ý: Cần chắc chắn dataframe có cột 'label_ground' và 'rule_label'
miss = df[(df['label_ground'] == 'Thịt - Hải sản - Trứng') & (df['rule_label'] == 'Khác')]
not_meat = df[df['label_ground'] != 'Thịt - Hải sản - Trứng']

caught = miss['item_name'].apply(match_meat_v3).sum()
fp = not_meat['item_name'].apply(match_meat_v3).sum()

print(f"✅ Recall : {caught}/{len(miss)} ({caught/max(len(miss),1)*100:.1f}%)")
print(f"✅ FP rate: {fp}/{len(not_meat)} ({fp/max(len(not_meat),1)*100:.2f}%)")

# ── 5. TRAIN MODEL MACHINE LEARNING ──
print("\nĐang train model...")
df['meat_v3'] = df['item_name'].apply(lambda x: int(match_meat_v3(x)))
df['enriched'] = df.apply(lambda r: r['clean'] + (' MEAT_V3' if r['meat_v3'] else ''), axis=1)

le = LabelEncoder()
y = le.fit_transform(df['label_ground'])
X_tr, X_te, y_tr, y_te = train_test_split(df['enriched'].fillna(''), y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(2,4), min_df=1, max_features=20000, sublinear_tf=True, analyzer='char_wb')),
    ('clf', LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced'))
])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
f1 = f1_score(y_te, y_pred, average='macro')

# ── 6. ĐÁNH GIÁ MODEL ──
print(f"\n✅ F1-macro (v3 final): {f1*100:.2f}%  (baseline: 97.59%)")
print("\n📋 Classification Report:")
print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))

f1_per = f1_score(y_te, y_pred, average=None)
print("Per-class F1:")
for cls, sc in sorted(zip(le.classes_, f1_per), key=lambda x: -x[1]):
    bar = '█'*int(sc*20) + '░'*(20-int(sc*20))
    print(f"  {cls:<35} {bar} {sc*100:5.1f}%")

# ── 7. EXPORT DATA (ĐÃ FIX LỖI PATH) ──
print("\nĐang lưu dữ liệu ra file csv...")
prob = pipe.predict_proba(df['enriched'].fillna(''))
pred = np.argmax(prob, axis=1)
conf = np.max(prob, axis=1)

df['ml_label_v3'] = le.inverse_transform(pred)
df['ml_conf_v3']  = np.round(conf, 3)
df['meat_rule_v3'] = df['meat_v3']

cols = ['category', 'user_id', 'product_group', 'item_name', 'clean', 'enriched',
        'label_ground', 'rule_label', 'rule_conf', 'ml_label_v3', 'ml_conf_v3', 'meat_rule_v3']

# Chỉ lấy những cột tồn tại trong dataframe thực tế để tránh lỗi KeyError
cols_to_export = [c for c in cols if c in df.columns]

# Sử dụng os.path.join để path được nối chính xác
df[cols_to_export].to_csv(os.path.join(OUT, 'foodfood_labeled_v3.csv'), index=False, encoding='utf-8-sig')

rules_rows = [{'label': 'Thịt - Hải sản - Trứng', 'type': 'meat_rule', 'pattern': p} for p in MEAT_RULES_V3]
excl_rows = [{'label': 'EXCLUSION', 'type': 'exclusion_prefix', 'pattern': p} for p in EXCLUSION_PREFIXES]
pd.DataFrame(rules_rows + excl_rows).to_csv(os.path.join(OUT, 'food_meat_rules_v3_final.csv'), index=False, encoding='utf-8-sig')

df_rft = df[df['ml_conf_v3'] >= 0.85][['item_name', 'clean', 'label_ground', 'ml_label_v3', 'ml_conf_v3']]
df_rft.to_csv(os.path.join(OUT, 'food_ready_for_train_v3.csv'), index=False, encoding='utf-8-sig')

print(f"""
╔═════════════════════════════════════════════════════╗
║      FOOD RULES V3 FINAL — SUMMARY                  ║
╠═════════════════════════════════════════════════════╣
║  Meat patterns          : {len(MEAT_RULES_V3):>4}                      ║
║  Exclusion prefixes     : {len(EXCLUSION_PREFIXES):>4}                      ║
║  Recall (miss→caught)   : {caught}/{len(miss)} ({caught/max(len(miss),1)*100:.1f}%)          ║
║  False positive rate    : {fp/max(len(not_meat),1)*100:>4.2f}%                     ║
║  F1-macro baseline      : 97.59%                    ║
║  F1-macro v3 final      : {f1*100:>5.2f}%                     ║
╚═════════════════════════════════════════════════════╝
""")
print(f"Đã lưu các file csv thành công tại: {OUT}")