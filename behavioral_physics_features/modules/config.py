"""
Behavioral Physics Feature Factory - Configuration
===================================================

Central configuration for all feature engineering modules.

Author: Behavioral Physics Team
Version: 1.0.0
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Time window configurations"""
    WINDOWS_MONTHS: List[int] = None
    WINDOWS_DAYS: List[int] = None

    def __post_init__(self):
        if self.WINDOWS_MONTHS is None:
            self.WINDOWS_MONTHS = [1, 3, 6, 12]
        if self.WINDOWS_DAYS is None:
            self.WINDOWS_DAYS = [30, 60, 90]


@dataclass
class StateConfig:
    """DPD state definitions (Thai simplified classification)"""
    # Simplified state boundaries (Thai regulatory classification)
    # CURRENT: 0-30 DPD, SM: 31-90 DPD, NPL: 91-180 DPD, CHARGE_OFF: 181+ DPD
    STATE_BOUNDARIES: Dict[str, Tuple[int, int]] = None

    # Regime definitions
    NORMAL_STATES: List[str] = None
    STRESSED_STATES: List[str] = None

    def __post_init__(self):
        if self.STATE_BOUNDARIES is None:
            # Thai simplified classification only
            self.STATE_BOUNDARIES = {
                "CURRENT": (0, 30),       # Current (0-30 DPD)
                "SM": (31, 90),           # Special Mention (31-90 DPD)
                "NPL": (91, 180),         # Non-Performing Loan (91-180 DPD)
                "CHARGE_OFF": (181, 99999)  # Charge-off / Loss (181+ DPD)
            }

        if self.NORMAL_STATES is None:
            self.NORMAL_STATES = ["CURRENT"]

        if self.STRESSED_STATES is None:
            self.STRESSED_STATES = ["SM", "NPL", "CHARGE_OFF"]

    def get_state_from_dpd(self, dpd: int) -> str:
        """Map DPD value to state"""
        for state, (lower, upper) in self.STATE_BOUNDARIES.items():
            if lower <= dpd <= upper:
                return state
        return "UNKNOWN"

    def is_normal_regime(self, state: str) -> bool:
        """Check if state is in NORMAL regime"""
        return state in self.NORMAL_STATES

    def is_stressed_regime(self, state: str) -> bool:
        """Check if state is in STRESSED regime"""
        return state in self.STRESSED_STATES


@dataclass
class LenderTypeConfig:
    """Lender type mapping configuration"""
    # Thailand-specific lender classification
    # Categories based on Thai financial institution types
    LENDER_TYPE_MAPPING: Dict[str, List[str]] = None

    # CardX identification
    CARDX_LENDER_IDS: List[str] = None

    def __post_init__(self):
        if self.LENDER_TYPE_MAPPING is None:
            # Map raw lender names/patterns to Thai financial institution types
            self.LENDER_TYPE_MAPPING = {
                "SFI": [
                    # Specialized Financial Institutions (สถาบันการเงินเฉพาะกิจ)
                    # State-owned financial institutions with specific mandates
                    "GOVERNMENT SAVINGS BANK",
                    "GSB",
                    "ธนาคารออมสิน",
                    "BANK FOR AGRICULTURE",
                    "BAAC",
                    "ธ.ก.ส",
                    "GOVERNMENT HOUSING BANK",
                    "GH BANK",
                    "ธนาคารอาคารสงเคราะห์",
                    "SME BANK",
                    "SME DEVELOPMENT BANK",
                    "ธนาคารพัฒนาวิสาหกิจขนาดกลางและขนาดย่อม",
                    "EXIM BANK",
                    "EXPORT-IMPORT BANK"
                ],
                "COMMERCIAL_BANK": [
                    # Commercial Banks (ธนาคารพาณิชย์)
                    # Thai and foreign commercial banks
                    "BANGKOK BANK",
                    "BBL",
                    "ธนาคารกรุงเทพ",
                    "KASIKORN",
                    "KBANK",
                    "K-BANK",
                    "ธนาคารกสิกรไทย",
                    "SCB",
                    "SIAM COMMERCIAL",
                    "ธนาคารไทยพาณิชย์",
                    "KRUNG THAI",
                    "KTB",
                    "ธนาคารกรุงไทย",
                    "TMB",
                    "THANACHART",
                    "TMB THANACHART",
                    "TTBBANK",
                    "ธนาคารทหารไทยธนชาต",
                    "KRUNGSRI",
                    "BAY",
                    "ธนาคารกรุงศรีอยุธยา",
                    "CIMB",
                    "CIMB THAI",
                    "UOB",
                    "UNITED OVERSEAS BANK",
                    "CITIBANK",
                    "CITI",
                    "HSBC",
                    "STANDARD CHARTERED",
                    "SC",
                    "TISCO",
                    "ธนาคารทิสโก้"
                ],
                "PERSONAL_LOAN": [
                    # Personal Loan Companies (บริษัทสินเชื่อส่วนบุคคล)
                    # Non-bank personal loan providers
                    "MUANG THAI",
                    "MUANGTHAI",
                    "มิวแทค",
                    "EASY BUY",
                    "อีซี่บาย",
                    "KRUNGSRI CONSUMER",
                    "FIRST CHOICE",
                    "เฟิร์สช้อยส์",
                    "MONEY TREE",
                    "PROMISE",
                    "พรอมิส",
                    "ACOM",
                    "AIFUL"
                ],
                "LEASING": [
                    # Leasing & Hire Purchase (บริษัทลิสซิ่ง)
                    # Auto leasing and hire purchase companies
                    "GE CAPITAL",
                    "TOYOTA LEASING",
                    "HONDA LEASING",
                    "NISSAN LEASING",
                    "ISUZU LEASING",
                    "KRUNGSRI AUTO",
                    "KRUNGTHAI LEASING",
                    "BANGKOK CAPITAL",
                    "SIAM CITY LEASING",
                    "THANACHART CAPITAL"
                ],
                "FINTECH": [
                    # Fintech & Digital Lenders (ฟินเทค)
                    # Digital lending platforms and BNPL
                    "RABBIT FINANCE",
                    "RABBIT LENDING",
                    "AEON",
                    "อิออน",
                    "MONIX",
                    "KREDIVO",
                    "ATOME",
                    "SHOPEE",
                    "SPAYLATER",
                    "LAZADA",
                    "LAZPAYLATER",
                    "GRAB",
                    "GRABPAY",
                    "TRUEMONEY",
                    "LINE BK",
                    "SCBX",
                    "FINNOMENA"
                ]
            }

        if self.CARDX_LENDER_IDS is None:
            # Replace with actual CardX lender IDs from your data
            self.CARDX_LENDER_IDS = ["CARDX", "CARDX_LENDER_ID"]

    def map_lender_type(self, lender_name_raw: str, lender_id: str) -> str:
        """
        Map raw lender name to Thai financial institution type.

        Returns: SFI, COMMERCIAL_BANK, PERSONAL_LOAN, LEASING, FINTECH, CARDX, OTHER

        Thai Categories:
        - SFI: Specialized Financial Institutions (สถาบันการเงินเฉพาะกิจ)
        - COMMERCIAL_BANK: Commercial Banks (ธนาคารพาณิชย์)
        - PERSONAL_LOAN: Personal Loan Companies (บริษัทสินเชื่อส่วนบุคคล)
        - LEASING: Leasing/Hire Purchase (บริษัทลิสซิ่ง)
        - FINTECH: Fintech/Digital Lenders (ฟินเทค)
        - CARDX: Internal CardX lender
        - OTHER: Unclassified
        """
        # Check if CardX
        if lender_id in self.CARDX_LENDER_IDS:
            return "CARDX"

        # Normalize lender name
        lender_upper = str(lender_name_raw).upper()

        # Check against mapping
        for lender_type, patterns in self.LENDER_TYPE_MAPPING.items():
            for pattern in patterns:
                if pattern in lender_upper:
                    return lender_type

        return "OTHER"


@dataclass
class ThresholdConfig:
    """Feature computation thresholds"""
    # Utilization thresholds
    UTIL_HIGH: float = 0.70
    UTIL_MAXED: float = 0.90

    # Payment thresholds
    PAYMENT_FULL: float = 0.95  # 95%+ of due = full payment
    PAYMENT_MIN: float = 0.30   # <30% of due = minimal payment

    # Enquiry thresholds
    ENQUIRY_HIGH_VELOCITY: int = 3  # 3+ enquiries in 3 months = high velocity
    ENQUIRY_BURST: int = 5          # 5+ in 1 month = burst

    # Cure thresholds
    CURE_IMPROVEMENT: int = 30  # Improvement of 30+ DPD = cure attempt

    # Transition speed thresholds (months)
    TRANSITION_FAST: int = 2    # S0→S2 in 2 months = fast deterioration
    TRANSITION_SLOW: int = 6    # S0→S2 in 6+ months = slow deterioration

    # Stickiness thresholds
    STICKY_MONTHS: int = 3      # 3+ consecutive months = sticky state

    # Outlier winsorization percentiles
    WINSORIZE_LOWER: float = 0.01
    WINSORIZE_UPPER: float = 0.99


@dataclass
class QualityConfig:
    """Data quality and validation thresholds"""
    # Missingness tolerance
    MAX_MISSINGNESS_PCT: float = 0.50  # 50% nulls = flag feature

    # Min observations per customer
    MIN_MONTHS_HISTORY: int = 3

    # Expected row counts
    EXPECTED_ROWS_PER_CUSTOMER: int = 1  # Per as_of_month

    # Leakage detection
    ENABLE_LEAKAGE_CHECKS: bool = True
    ENABLE_FUTURE_DATA_CHECKS: bool = True


# ============================================================================
# Global Configuration Instance
# ============================================================================

class BehavioralPhysicsConfig:
    """Master configuration object"""

    def __init__(self):
        self.windows = WindowConfig()
        self.states = StateConfig()
        self.lender_types = LenderTypeConfig()
        self.thresholds = ThresholdConfig()
        self.quality = QualityConfig()

    def to_dict(self) -> Dict:
        """Export config as dictionary"""
        return {
            "windows": {
                "months": self.windows.WINDOWS_MONTHS,
                "days": self.windows.WINDOWS_DAYS
            },
            "states": {
                "boundaries": self.states.STATE_BOUNDARIES,
                "normal": self.states.NORMAL_STATES,
                "stressed": self.states.STRESSED_STATES
            },
            "thresholds": {
                "utilization_high": self.thresholds.UTIL_HIGH,
                "utilization_maxed": self.thresholds.UTIL_MAXED,
                "payment_full": self.thresholds.PAYMENT_FULL,
                "enquiry_burst": self.thresholds.ENQUIRY_BURST
            }
        }


# Create global config instance
CONFIG = BehavioralPhysicsConfig()


# ============================================================================
# Helper Functions
# ============================================================================

def get_config() -> BehavioralPhysicsConfig:
    """Get global configuration instance"""
    return CONFIG


def update_cardx_lender_ids(lender_ids: List[str]):
    """Update CardX lender IDs (for production deployment)"""
    CONFIG.lender_types.CARDX_LENDER_IDS = lender_ids


def add_window(window_months: int):
    """Add custom window to configuration"""
    if window_months not in CONFIG.windows.WINDOWS_MONTHS:
        CONFIG.windows.WINDOWS_MONTHS.append(window_months)
        CONFIG.windows.WINDOWS_MONTHS.sort()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Behavioral Physics Configuration")
    print("=" * 70)

    config = get_config()

    print("\nState Boundaries:")
    for state, (lower, upper) in config.states.STATE_BOUNDARIES.items():
        print(f"  {state}: {lower}-{upper} DPD")

    print("\nRegimes:")
    print(f"  NORMAL: {config.states.NORMAL_STATES}")
    print(f"  STRESSED: {config.states.STRESSED_STATES}")

    print("\nWindows:")
    print(f"  Months: {config.windows.WINDOWS_MONTHS}")
    print(f"  Days: {config.windows.WINDOWS_DAYS}")

    print("\nLender Types:")
    for ltype, patterns in config.lender_types.LENDER_TYPE_MAPPING.items():
        print(f"  {ltype}: {len(patterns)} patterns")

    # Test state mapping
    print("\nState Mapping Tests:")
    test_dpds = [0, 15, 45, 120, 200]
    for dpd in test_dpds:
        state = config.states.get_state_from_dpd(dpd)
        regime = "NORMAL" if config.states.is_normal_regime(state) else "STRESSED"
        print(f"  DPD {dpd:3d} → {state} ({regime})")

    # Test lender mapping
    print("\nLender Type Mapping Tests:")
    test_lenders = [
        ("BANGKOK BANK", "BB001"),
        ("RABBIT FINANCE", "RF001"),
        ("CARDX", "CARDX"),
        ("UNKNOWN LENDER", "UNK001")
    ]
    for lender_name, lender_id in test_lenders:
        lender_type = config.lender_types.map_lender_type(lender_name, lender_id)
        print(f"  {lender_name:20s} → {lender_type}")
