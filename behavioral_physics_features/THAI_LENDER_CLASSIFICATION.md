# Thai Lender Classification Guide

**Purpose**: Map Thai lenders from bureau data to standardized financial institution types

**File**: `behavioral_physics_features/modules/config.py` (LenderTypeConfig class)

---

## 🇹🇭 Thai Financial Institution Categories

### 1. SFI (Specialized Financial Institutions)
**Thai**: สถาบันการเงินเฉพาะกิจ

**Description**: State-owned financial institutions with specific mandates (agriculture, housing, SMEs, etc.)

**Examples**:
- Government Savings Bank (GSB / ธนาคารออมสิน)
- Bank for Agriculture and Agricultural Cooperatives (BAAC / ธ.ก.ส.)
- Government Housing Bank (GH Bank / ธนาคารอาคารสงเคราะห์)
- SME Development Bank (ธนาคารพัฒนาวิสาหกิจขนาดกลางและขนาดย่อม)
- Export-Import Bank of Thailand (EXIM Bank)

**Characteristics**:
- Government-owned
- Policy-driven lending
- Specific target segments
- Often lower interest rates

---

### 2. COMMERCIAL_BANK (Commercial Banks)
**Thai**: ธนาคารพาณิชย์

**Description**: Full-service commercial banks (Thai and foreign)

**Examples**:
- **Thai banks**: Bangkok Bank (BBL), Kasikorn (KBANK), Siam Commercial Bank (SCB), Krung Thai (KTB), TMB Thanachart (TTB), Krungsri (BAY), TISCO
- **Foreign banks**: CIMB Thai, UOB, Citibank, HSBC, Standard Chartered

**Characteristics**:
- Full banking license
- Regulated by Bank of Thailand
- Deposit-taking institutions
- Offer full range of financial products

---

### 3. PERSONAL_LOAN (Personal Loan Companies)
**Thai**: บริษัทสินเชื่อส่วนบุคคล

**Description**: Non-bank consumer finance companies focusing on unsecured personal loans

**Examples**:
- Muang Thai Leasing (มิวแทค)
- Easy Buy (อีซี่บาย)
- Krungsri Consumer
- First Choice (เฟิร์สช้อยส์)
- Money Tree
- Promise (พรอมิส)
- ACOM, AIFUL

**Characteristics**:
- Non-bank lenders
- Focus on unsecured personal loans
- Higher interest rates than banks
- Often target salaried workers
- Cash loans, debt consolidation

---

### 4. LEASING (Leasing & Hire Purchase)
**Thai**: บริษัทลิสซิ่ง / บริษัทเช่าซื้อ

**Description**: Companies specializing in auto leasing and hire purchase

**Examples**:
- GE Capital
- Toyota Leasing
- Honda Leasing
- Nissan Leasing
- Isuzu Leasing
- Krungsri Auto
- Krungthai Leasing
- Bangkok Capital
- Siam City Leasing
- Thanachart Capital

**Characteristics**:
- Secured lending (vehicle as collateral)
- Auto financing specialists
- Often affiliated with car manufacturers
- Hire purchase agreements

---

### 5. FINTECH (Fintech & Digital Lenders)
**Thai**: ฟินเทค / ผู้ให้บริการสินเชื่อดิจิทัล

**Description**: Digital lending platforms and Buy Now Pay Later (BNPL) providers

**Examples**:
- Rabbit Finance / Rabbit Lending
- AEON (อิออน)
- Monix
- Kredivo
- Atome
- Shopee (SPayLater)
- Lazada (LazPayLater)
- Grab (GrabPay)
- TrueMoney
- LINE BK
- SCBX

**Characteristics**:
- Digital-first/app-based
- Fast approval process
- Often BNPL (Buy Now Pay Later)
- E-commerce integrated
- Smaller ticket sizes
- Tech-driven underwriting

---

### 6. CARDX
**Description**: Internal CardX lender

**Identification**: By lender_id matching CARDX_LENDER_IDS list

---

### 7. OTHER
**Description**: Unclassified lenders not matching any category

**Action Required**: Review "OTHER" lenders periodically and add to appropriate category

---

## 🔍 How Classification Works

### Matching Logic
```python
def map_lender_type(lender_name_raw: str, lender_id: str) -> str:
    """
    1. Check if lender_id matches CARDX → return "CARDX"
    2. Uppercase the lender_name_raw
    3. Check if any pattern in category lists appears in lender name
    4. Return first matching category
    5. If no match, return "OTHER"
    """
```

### Example Matches
| Bureau Name | Matched Pattern | Category |
|-------------|-----------------|----------|
| "ธนาคารกรุงเทพ จำกัด (มหาชน)" | "ธนาคารกรุงเทพ" | COMMERCIAL_BANK |
| "GOVERNMENT SAVINGS BANK" | "GOVERNMENT SAVINGS BANK" | SFI |
| "Muang Thai Capital PCL" | "MUANG THAI" | PERSONAL_LOAN |
| "TOYOTA LEASING (THAILAND) CO., LTD." | "TOYOTA LEASING" | LEASING |
| "SHOPEE PAYLATER" | "SHOPEE" | FINTECH |

---

## 📊 Validating Lender Classification

### Step 1: Check Actual Bureau Lender Names

Run this SQL in Databricks:

```sql
-- Get all unique lender names from bureau
SELECT
    MEMBERSHORTNAME as lender_name,
    MEMBERCODE as lender_id,
    COUNT(DISTINCT REF_NO) as customer_count
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE MEMBERSHORTNAME IS NOT NULL
GROUP BY 1, 2
ORDER BY 3 DESC
LIMIT 100;
```

### Step 2: Test Classification

```python
from behavioral_physics_features.modules.config import LenderTypeConfig

config = LenderTypeConfig()

# Test actual lender names
test_lenders = [
    ("ธนาคารกรุงเทพ จำกัด (มหาชน)", "BBL001"),
    ("บมจ. กสิกรไทย", "KBANK001"),
    ("GOVERNMENT SAVINGS BANK", "GSB001"),
    ("Muang Thai Capital PCL", "MT001"),
    # Add more from your actual data
]

for name, id in test_lenders:
    lender_type = config.map_lender_type(name, id)
    print(f"{name:<50} → {lender_type}")
```

### Step 3: Find Unclassified Lenders

```python
# After running pipeline
bureau_trade.filter(F.col("lender_type") == "OTHER") \
    .select("lender_name", "lender_id") \
    .distinct() \
    .show(50, truncate=False)
```

**Action**: Add these to appropriate category in config.py

---

## 🛠️ Updating Classification

### Adding New Lenders

1. **Identify the lender type** (SFI, Commercial Bank, etc.)

2. **Add to config.py**:
```python
LENDER_TYPE_MAPPING = {
    "COMMERCIAL_BANK": [
        ...,
        "NEW BANK NAME",  # Add here
        "ธนาคารใหม่",      # Thai name if applicable
    ],
}
```

3. **Add both English and Thai names** if available

4. **Add abbreviations** (e.g., "KBANK", "K-BANK", "KASIKORN")

5. **Test the mapping**:
```python
config = LenderTypeConfig()
assert config.map_lender_type("NEW BANK NAME", "ID") == "COMMERCIAL_BANK"
```

### Pattern Matching Tips

**Good patterns** (specific enough to avoid false matches):
- ✅ "BANGKOK BANK" (specific)
- ✅ "ธนาคารกรุงเทพ" (specific)
- ✅ "KASIKORN" (specific)

**Bad patterns** (too generic):
- ❌ "BANK" (matches everything)
- ❌ "FINANCE" (matches everything)
- ❌ "CAPITAL" (too generic)

---

## 📈 Feature Impact by Category

### Balance/Limit Share Features
- `sfi_balance_share`: % of total balance with SFI lenders
- `commercial_bank_balance_share`: % with commercial banks
- `personal_loan_balance_share`: % with personal loan companies
- `leasing_balance_share`: % with leasing companies
- `fintech_balance_share`: % with fintech lenders
- `cardx_balance_share`: % with CardX

### Category-Specific Insights

| Category | Typical Customer Profile | Risk Characteristics |
|----------|-------------------------|---------------------|
| SFI | Lower income, rural, specific segments | Lower default rates (government backing) |
| COMMERCIAL_BANK | Salaried, middle-upper income | Medium risk, well-monitored |
| PERSONAL_LOAN | Salaried, need quick cash | Higher interest, higher risk |
| LEASING | Auto buyers | Secured by vehicle, medium risk |
| FINTECH | Young, digital-savvy | Variable risk, newer segment |
| CARDX | Internal customers | Known risk profile |

---

## 🚨 Common Issues

### Issue 1: Lender classified as "OTHER" but should be classified

**Cause**: Lender name doesn't match any pattern

**Fix**: Add lender name pattern to appropriate category in config.py

### Issue 2: Wrong classification

**Cause**: Pattern too generic (e.g., "CAPITAL" matches both "Bangkok Capital" and "Thanachart Capital")

**Fix**: Use more specific patterns, check order of matching

### Issue 3: Thai characters not matching

**Cause**: Encoding issues or incomplete patterns

**Fix**: Add both Thai and English names to pattern list

---

## ✅ Validation Checklist

- [ ] Check top 20 lenders by volume are correctly classified
- [ ] "OTHER" category should be <5% of total accounts
- [ ] Each category has reasonable distribution (not 0% or 100%)
- [ ] Thai bank names match correctly
- [ ] Foreign bank names match correctly
- [ ] Fintech lenders properly identified
- [ ] SFI lenders separated from commercial banks
- [ ] Personal loan companies not confused with banks

---

## 📞 Next Steps

1. **Run validation query** to see actual lender names
2. **Test classification** on top 50 lenders
3. **Add missing patterns** to config.py
4. **Re-run pipeline** to verify features
5. **Monitor "OTHER" category** - should be minimal

---

**File to modify**: `behavioral_physics_features/modules/config.py` (LenderTypeConfig class)

**Features affected**: All lender ecology features (~25 features)
