# CardX — Graph-Based Transaction Fraud Detection: POC Brief

**Version:** 1.0 | **Date:** February 2026 | **Status:** POC Proposal

---

## 1. Executive Summary

Traditional rule-based and tabular ML fraud models treat each transaction in isolation. They miss **relationship-based fraud signals** — fraud rings, shared devices, recycled accounts, and coordinated attacks — that only become visible when you model **who transacts with whom, using what, from where**.

This POC demonstrates how a **Graph Fraud Model** built on CardX transaction data can:

- Detect **fraud rings** coordinating across multiple cards and merchants
- Surface **account takeover** signals via device/IP anomalies
- **Propagate risk** from confirmed fraud nodes to connected entities
- Improve **precision at low false-positive thresholds** — critical for a credit card company

---

## 2. The Problem: Why Graph Matters for CardX

### Limitations of Tabular-Only Fraud Models

| Signal Type | Tabular Model | Graph Model |
|-------------|--------------|-------------|
| High-value single transaction | ✅ Captured | ✅ Captured |
| Velocity (card-level) | ✅ Captured | ✅ Captured |
| Shared device across 10 cards | ❌ Missed | ✅ Captured |
| Fraud ring of 50 coordinated accounts | ❌ Missed | ✅ Captured |
| Merchant colluding with cardholders | ❌ Missed | ✅ Captured |
| Risk propagated from known fraudster | ❌ Missed | ✅ Captured |
| "Mule" account bridging rings | ❌ Missed | ✅ Captured |

### Core Insight

> **Fraud rarely acts alone.** A stolen card is tested at a specific merchant. Multiple stolen cards share a device. A compromised merchant touches hundreds of legitimate cards. Graph models expose these hidden connections.

---

## 3. CardX Transaction Data: Tables & Columns

### 3.1 Table: `transactions`

The central fact table. Every swipe, tap, or online charge.

| Column | Type | Description | Graph Role |
|--------|------|-------------|------------|
| `transaction_id` | STRING (PK) | Unique transaction identifier | Edge ID |
| `card_id` | STRING (FK) | Card used for this transaction | Node → Card |
| `merchant_id` | STRING (FK) | Merchant where transaction occurred | Node → Merchant |
| `device_id` | STRING (FK) | Device fingerprint at time of transaction | Node → Device |
| `ip_address` | STRING | IP address of transaction origin | Node → IP |
| `transaction_timestamp` | TIMESTAMP | Date and time of transaction | Temporal edge weight |
| `transaction_amount` | DECIMAL(12,2) | Transaction value in USD | Edge weight |
| `transaction_type` | STRING | `purchase`, `refund`, `cash_advance`, `transfer` | Edge type |
| `mcc_code` | STRING | Merchant Category Code (e.g., 5411=Grocery) | Feature |
| `channel` | STRING | `in_store`, `online`, `mobile`, `atm` | Feature |
| `currency` | STRING | Transaction currency (ISO 4217) | Feature |
| `authorization_code` | STRING | Card network authorization code | Feature |
| `response_code` | STRING | `approved`, `declined`, `reversed` | Feature |
| `is_international` | BOOLEAN | Whether transaction is cross-border | Feature |
| `is_fraud` | BOOLEAN | **Fraud label (ground truth)** | Node/Edge label |
| `fraud_type` | STRING | `card_testing`, `account_takeover`, `friendly_fraud`, `ring_fraud` | Label detail |

---

### 3.2 Table: `cards`

One row per card. Links cardholders to the payment instruments they use.

| Column | Type | Description | Graph Role |
|--------|------|-------------|------------|
| `card_id` | STRING (PK) | Unique card identifier | Node |
| `customer_id` | STRING (FK) | Owning cardholder | Edge: Card → Customer |
| `card_type` | STRING | `credit`, `debit`, `prepaid` | Node attribute |
| `card_network` | STRING | `visa`, `mastercard`, `amex`, `discover` | Node attribute |
| `card_status` | STRING | `active`, `suspended`, `closed`, `compromised` | Risk attribute |
| `issue_date` | DATE | Card issue date | Temporal feature |
| `expiry_date` | DATE | Card expiry | Feature |
| `credit_limit` | DECIMAL(10,2) | Approved credit limit | Feature |
| `current_balance` | DECIMAL(10,2) | Outstanding balance | Feature |
| `days_since_last_fraud` | INT | Days since card was last flagged | Risk feature |

---

### 3.3 Table: `customers`

Cardholder profile. A customer may have multiple cards.

| Column | Type | Description | Graph Role |
|--------|------|-------------|------------|
| `customer_id` | STRING (PK) | Unique customer identifier | Node |
| `email_hash` | STRING | Hashed email address | Shared-identity edge |
| `phone_hash` | STRING | Hashed phone number | Shared-identity edge |
| `zip_code` | STRING | Customer's registered ZIP | Feature |
| `state` | STRING | State of residence | Feature |
| `customer_since_date` | DATE | Account opening date | Temporal feature |
| `kyc_status` | STRING | `verified`, `pending`, `failed` | Risk attribute |
| `account_status` | STRING | `active`, `frozen`, `closed` | Node attribute |
| `fico_score` | INT | Credit score (300–850) | Risk feature |
| `num_cards` | INT | Total cards held | Feature |

---

### 3.4 Table: `merchants`

One row per merchant. Merchants are key hub nodes — a compromised merchant touches thousands of cards.

| Column | Type | Description | Graph Role |
|--------|------|-------------|------------|
| `merchant_id` | STRING (PK) | Unique merchant identifier | Node |
| `merchant_name` | STRING | Business name | Node label |
| `mcc_code` | STRING | Merchant Category Code | Node attribute |
| `merchant_category` | STRING | `grocery`, `gas`, `electronics`, `online_retail` | Node attribute |
| `city` | STRING | Merchant city | Feature |
| `state` | STRING | Merchant state | Feature |
| `country` | STRING | Merchant country (ISO 3166) | Feature |
| `is_online` | BOOLEAN | Online-only merchant | Feature |
| `fraud_rate_30d` | FLOAT | 30-day rolling fraud rate at this merchant | Risk attribute |
| `chargeback_rate_90d` | FLOAT | 90-day chargeback rate | Risk attribute |
| `merchant_risk_tier` | STRING | `low`, `medium`, `high`, `critical` | Risk label |

---

### 3.5 Table: `devices`

Device fingerprints associated with transactions. Shared devices are the strongest fraud signal.

| Column | Type | Description | Graph Role |
|--------|------|-------------|------------|
| `device_id` | STRING (PK) | Unique device fingerprint | Node |
| `device_type` | STRING | `mobile`, `desktop`, `tablet`, `pos_terminal` | Node attribute |
| `os` | STRING | Operating system | Feature |
| `browser` | STRING | Browser fingerprint (for online txns) | Feature |
| `first_seen_date` | DATE | First time this device was seen | Temporal feature |
| `num_cards_used` | INT | Count of distinct cards used on this device | **Key fraud signal** |
| `num_customers` | INT | Count of distinct customers using this device | **Key fraud signal** |
| `is_vpn` | BOOLEAN | Whether VPN/proxy was detected | Risk attribute |
| `is_emulator` | BOOLEAN | Whether device appears to be an emulator | Risk attribute |

---

### 3.6 Table: `ip_addresses`

IP-level metadata. Useful for geolocation inconsistency and shared IP fraud.

| Column | Type | Description | Graph Role |
|--------|------|-------------|------------|
| `ip_id` | STRING (PK) | Unique IP identifier | Node |
| `ip_address` | STRING | Raw IP address (hashed for PII) | Node key |
| `country` | STRING | Country of origin | Feature |
| `city` | STRING | City of origin | Feature |
| `isp` | STRING | Internet service provider | Feature |
| `is_tor` | BOOLEAN | Whether IP is a Tor exit node | Risk attribute |
| `is_datacenter` | BOOLEAN | Whether IP is from a datacenter | Risk attribute |
| `abuse_score` | FLOAT (0–1) | IP reputation score | Risk attribute |
| `num_cards_seen` | INT | Count of distinct cards from this IP | **Key fraud signal** |

---

## 4. Graph Data Model for Fraud Detection

### 4.1 Node Types (Entities)

```
CUSTOMER   ──── represents a cardholder
CARD       ──── a payment instrument (may be stolen)
MERCHANT   ──── where transactions occur (may be compromised)
DEVICE     ──── the hardware/browser used (shared = red flag)
IP_ADDRESS ──── network origin (shared = red flag)
```

### 4.2 Edge Types (Relationships)

```
CUSTOMER  ──[OWNS]──────────► CARD
CARD      ──[USED_AT]───────► MERCHANT      (weight = transaction_amount, time = timestamp)
CARD      ──[USED_FROM]─────► DEVICE        (shared device = fraud ring signal)
CARD      ──[ACCESSED_VIA]──► IP_ADDRESS    (shared IP = coordinated attack signal)
CUSTOMER  ──[SHARES_DEVICE]─► CUSTOMER      (derived — same device, different cardholders)
CUSTOMER  ──[SHARES_IP]─────► CUSTOMER      (derived — same IP, different cardholders)
MERCHANT  ──[CONNECTED_TO]──► MERCHANT      (derived — same cards transacted at both)
```

### 4.3 Sample Graph Topology

```
                  [DEVICE_A]
                  /   |   \
[CARD_001]──────/    |    \──── [CARD_007]   ← FRAUD RING: 3 cards share same device
[CARD_003]──────     |
[CARD_007]──────     └──── [IP_192.168.5.1]
     |
     ├──[TXN_100]──► [MERCHANT_GAS_STATION]  ← Card-testing pattern
     ├──[TXN_101]──► [MERCHANT_GAS_STATION]    (small amounts, same merchant)
     └──[TXN_102]──► [MERCHANT_ELECTRONICS]  ← Large fraudulent purchase follows
```

---

## 5. Graph Fraud Modeling Approach for the POC

### 5.1 Phase 1 — Graph Construction

**Input:** Raw transaction tables (above)
**Output:** A property graph with ~5 node types and ~7 edge types

```
Transactions → Card–Merchant edges
Cards        → Card nodes + Customer–Card edges
Devices      → Device nodes + Card–Device edges
IP addresses → IP nodes + Card–IP edges
```

### 5.2 Phase 2 — Graph Feature Extraction

For each **Card** node (scoring unit), extract:

| Feature Group | Features | Fraud Signal |
|---------------|----------|--------------|
| **Centrality** | Degree, PageRank, Betweenness | High-degree card = hub in fraud ring |
| **Structural** | Clustering coefficient, k-core number | Dense subgraph = organized fraud |
| **Community** | Louvain community ID, community size | Large community = fraud syndicate |
| **Shared Identity** | # cards sharing same device/IP | Direct fraud ring indicator |
| **Risk Propagation** | Propagated risk from known fraud nodes | Guilt by association |
| **Merchant Risk** | Avg merchant fraud rate in neighborhood | Connected to bad merchants |
| **Temporal** | Burst of edges in short time window | Card-testing velocity |

### 5.3 Phase 3 — Fraud Detection Models

**Layer 1 — Rule-Based Thresholds on Graph Features:**
- `num_cards_on_same_device > 5` → Flag
- `propagated_risk_score > 0.7` → Flag
- `community_size > 20 AND known_fraud_in_community > 0` → Flag

**Layer 2 — ML Model on Graph + Tabular Features:**
- XGBoost / LightGBM with graph features concatenated to tabular features
- Expected lift: +15–25% AUC vs. tabular-only baseline

**Layer 3 — Graph Neural Network (future):**
- GraphSAGE or GAT for inductive inference on new nodes
- Captures structural patterns that hand-crafted features miss

### 5.4 Key Fraud Patterns the Graph Exposes

#### Pattern 1: Fraud Ring Detection
```
Symptom: Multiple cards share the same device_id or IP
Graph signal: High clustering coefficient in Card–Device bipartite graph
Action: Flag all cards in the cluster for review
```

#### Pattern 2: Card Testing
```
Symptom: Multiple small transactions at a single merchant, followed by a large one
Graph signal: High edge velocity (Card→Merchant) in short time window
Action: Decline large transaction, trigger step-up authentication
```

#### Pattern 3: Account Takeover (ATO)
```
Symptom: Card suddenly used from a new device/IP inconsistent with history
Graph signal: New edge to a Device node with high num_cards_seen
Action: Real-time challenge (OTP/biometric)
```

#### Pattern 4: Merchant Compromise
```
Symptom: Merchant node suddenly connects to many high-risk IP nodes
Graph signal: Merchant PageRank spikes; connected cards' risk propagates
Action: Isolate merchant; proactively reissue affected cards
```

#### Pattern 5: Mule Account Networks
```
Symptom: A card receives funds and immediately forwards to another card at same merchant
Graph signal: Directed flow graph shows "bridge" nodes
Action: SAR (Suspicious Activity Report) filing
```

---

## 6. Data Access Requirements for the POC

### 6.1 Minimum Data Needed

| Table | Rows Needed | Time Window |
|-------|-------------|-------------|
| `transactions` | 5M–50M rows | 6 months |
| `cards` | 500K–5M | Full history |
| `customers` | 200K–2M | Full history |
| `merchants` | 10K–100K | Full history |
| `devices` | 1M–10M | 6 months |
| `ip_addresses` | 500K–5M | 6 months |

### 6.2 Fraud Label Requirements
- At minimum: `is_fraud` flag on `transactions`
- Ideally: `fraud_type` to train type-specific models
- Acceptable fraud rate: 0.1%–2% (class imbalance handling built-in)

### 6.3 Privacy & Compliance
- PII columns (`email`, `phone`, `ip_address`) should be **hashed or tokenized**
- Device IDs should use **pseudonymous fingerprints** (not raw browser data)
- All data access governed by CardX data governance framework

---

## 7. Expected POC Outcomes

| Metric | Baseline (Tabular) | Expected with Graph |
|--------|--------------------|---------------------|
| AUC-ROC | ~0.85 | ~0.90–0.93 |
| Precision @ 10% recall | ~60% | ~72–78% |
| Fraud ring detection rate | ~20% | ~70–85% |
| False positive rate | Baseline | ~20–30% reduction |
| New fraud types detected | 0 | 2–3 novel patterns |

---

## 8. Technology Stack for POC

| Component | Technology |
|-----------|------------|
| Data storage | Delta Lake (Databricks) |
| Graph construction | NetworkX (POC) → GraphX / Neo4j (production) |
| Graph features | `GraphFeatureEngine` (existing platform module) |
| ML model | LightGBM + graph features |
| Orchestration | Databricks Workflows |
| Visualization | Gephi / PyVis / Plotly for graph visualization |
| Experiment tracking | MLflow |

---

## 9. POC Deliverables & Timeline

| Week | Deliverable |
|------|-------------|
| 1 | Data access confirmed; schema mapping validated |
| 2 | Graph construction pipeline; baseline tabular model |
| 3 | Graph features extracted; fraud pattern rules |
| 4 | Graph-enhanced ML model; AUC comparison |
| 5 | Fraud ring visualization; stakeholder demo |

---

*This brief was prepared as part of the CardX Graph Fraud Modeling POC initiative.*
