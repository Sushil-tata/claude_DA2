-- Quick SQL validation of XXX in payment history data
-- Run this in Databricks SQL for fast checking

-- ============================================================================
-- 1. Check how many accounts have XXX in payment history
-- ============================================================================
SELECT
    COUNT(*) as total_accounts,
    SUM(CASE WHEN PAYMENTHISTORY1 LIKE '%XXX%' THEN 1 ELSE 0 END) as accounts_with_xxx,
    ROUND(SUM(CASE WHEN PAYMENTHISTORY1 LIKE '%XXX%' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as pct_with_xxx
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 IS NOT NULL;

-- ============================================================================
-- 2. Sample payment history strings with XXX
-- ============================================================================
SELECT
    REF_NO,
    ACCOUNTNUMBER,
    MEMBERSHORTNAME,
    PAYMENTHISTORY1,
    LENGTH(PAYMENTHISTORY1) as ph_length,
    -- Count total XXX occurrences
    (LENGTH(PAYMENTHISTORY1) - LENGTH(REPLACE(PAYMENTHISTORY1, 'XXX', ''))) / 3 as total_xxx_count,
    -- Get last 36 chars (last 12 months)
    RIGHT(PAYMENTHISTORY1, 36) as last_12_months
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 LIKE '%XXX%'
LIMIT 20;

-- ============================================================================
-- 3. Distribution of XXX in different positions
-- ============================================================================
SELECT
    CASE
        WHEN PAYMENTHISTORY1 LIKE 'XXX%' THEN 'XXX at start'
        WHEN PAYMENTHISTORY1 LIKE '%XXX' THEN 'XXX at end'
        WHEN PAYMENTHISTORY1 LIKE '%XXX%' THEN 'XXX in middle'
        ELSE 'No XXX'
    END as xxx_position,
    COUNT(*) as account_count
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC;

-- ============================================================================
-- 4. Check for accounts with ALL XXX (completely not reported)
-- ============================================================================
SELECT
    COUNT(*) as all_xxx_accounts
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 IS NOT NULL
  AND REPLACE(PAYMENTHISTORY1, 'XXX', '') = '';  -- All XXX when we remove them

-- ============================================================================
-- 5. Sample accounts: No XXX vs Some XXX vs All XXX
-- ============================================================================
-- No XXX (clean reporting)
SELECT 'NO_XXX' as category, REF_NO, ACCOUNTNUMBER, PAYMENTHISTORY1
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 IS NOT NULL
  AND PAYMENTHISTORY1 NOT LIKE '%XXX%'
LIMIT 5

UNION ALL

-- Some XXX (partial gaps)
SELECT 'SOME_XXX' as category, REF_NO, ACCOUNTNUMBER, PAYMENTHISTORY1
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 LIKE '%XXX%'
  AND REPLACE(PAYMENTHISTORY1, 'XXX', '') != ''
LIMIT 5

UNION ALL

-- All XXX (complete gap)
SELECT 'ALL_XXX' as category, REF_NO, ACCOUNTNUMBER, PAYMENTHISTORY1
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 IS NOT NULL
  AND REPLACE(PAYMENTHISTORY1, 'XXX', '') = ''
LIMIT 5;

-- ============================================================================
-- 6. Check last 12 months specifically (last 36 characters)
-- ============================================================================
SELECT
    CASE
        WHEN RIGHT(PAYMENTHISTORY1, 36) LIKE '%XXX%' THEN 'Has XXX in last 12 months'
        ELSE 'No XXX in last 12 months'
    END as recent_xxx,
    COUNT(*) as account_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
FROM cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account
WHERE PAYMENTHISTORY1 IS NOT NULL
  AND LENGTH(PAYMENTHISTORY1) >= 36
GROUP BY 1
ORDER BY 2 DESC;
