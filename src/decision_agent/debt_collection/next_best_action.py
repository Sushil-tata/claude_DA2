"""
Next Best Action & Next Best Offer Decision Engine

Recommends optimal channel, timing, and offer for each debtor
based on persona, propensity scores, and uplift models.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, functions as F
import mlflow


class NextBestActionEngine:
    """
    Determines optimal collection action for each account:
    - Which channel to use (email, SMS, call, voice bot)
    - When to contact (time of day, day of week)
    - How often to contact (frequency optimization)
    - Compliance checks
    """

    def __init__(self, config: Dict):
        self.config = config
        self.channel_strategy = config['next_best_action']['channel_strategy']
        self.timing_strategy = config['next_best_action']['timing_strategy']
        self.constraints = config['constraints']

    def recommend_action(
        self,
        account_features: Dict,
        persona: str,
        model_scores: Dict,
        contact_history: pd.DataFrame
    ) -> Dict:
        """
        Recommend next best action for an account.

        Args:
            account_features: Account characteristics
            persona: Assigned persona/segment
            model_scores: {
                'ltp_score': 0.65,
                'atp_score': 0.50,
                'channel_probabilities': {'email': 0.2, 'sms': 0.5, 'call': 0.3}
            }
            contact_history: Recent contact history for this account

        Returns:
            {
                'recommended_channel': 'sms',
                'recommended_time': '2024-02-10 14:00:00',
                'confidence': 0.75,
                'compliant': True,
                'reason': 'High SMS response rate for this persona'
            }
        """

        # Step 1: Check compliance constraints
        compliance_check = self._check_compliance(
            account_features, contact_history
        )

        if not compliance_check['compliant']:
            return {
                'recommended_channel': 'none',
                'recommended_time': None,
                'confidence': 1.0,
                'compliant': False,
                'reason': compliance_check['reason']
            }

        # Step 2: Determine channel
        channel = self._select_channel(
            account_features, persona, model_scores, contact_history
        )

        # Step 3: Determine timing
        contact_time = self._select_timing(
            account_features, persona, model_scores, contact_history
        )

        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(
            channel, model_scores
        )

        return {
            'recommended_channel': channel['channel'],
            'recommended_time': contact_time,
            'confidence': confidence,
            'compliant': True,
            'reason': channel['reason'],
            'expected_response_rate': channel['expected_response_rate']
        }

    def _check_compliance(
        self,
        account_features: Dict,
        contact_history: pd.DataFrame
    ) -> Dict:
        """Check regulatory compliance constraints"""

        # Check cease and desist
        if account_features.get('cease_and_desist_flag', False):
            return {
                'compliant': False,
                'reason': 'Cease and desist flag set - no contact allowed'
            }

        # Check max contacts per week
        recent_contacts = contact_history[
            contact_history['contact_date'] >= datetime.now() - timedelta(days=7)
        ]

        max_contacts = self.constraints['regulatory']['max_contact_attempts_per_week']
        if len(recent_contacts) >= max_contacts:
            return {
                'compliant': False,
                'reason': f'Exceeded max contacts per week ({max_contacts})'
            }

        # Check dispute status
        if account_features.get('dispute_rate', 0) > 0.3:
            return {
                'compliant': False,
                'reason': 'Active dispute - requires compliance review'
            }

        return {'compliant': True, 'reason': 'All compliance checks passed'}

    def _select_channel(
        self,
        account_features: Dict,
        persona: str,
        model_scores: Dict,
        contact_history: pd.DataFrame
    ) -> Dict:
        """Select optimal contact channel"""

        # Apply rule-based channel selection first
        for rule in self.channel_strategy['rules']:
            if self._evaluate_rule(rule['if'], account_features, persona):
                if rule['then'] == 'use_model_prediction':
                    # Use ML model
                    break
                else:
                    # Use rule-based channel
                    return {
                        'channel': rule['then'],
                        'reason': f'Rule-based: {rule["if"]}',
                        'expected_response_rate': 0.3  # Default estimate
                    }

        # Use model prediction
        channel_probs = model_scores.get('channel_probabilities', {})

        if not channel_probs:
            # Fallback to historical best channel
            return self._historical_best_channel(contact_history)

        # Select channel with highest probability
        best_channel = max(channel_probs.items(), key=lambda x: x[1])

        return {
            'channel': best_channel[0],
            'reason': f'Model prediction (confidence: {best_channel[1]:.2f})',
            'expected_response_rate': best_channel[1]
        }

    def _select_timing(
        self,
        account_features: Dict,
        persona: str,
        model_scores: Dict,
        contact_history: pd.DataFrame
    ) -> datetime:
        """Select optimal contact timing"""

        # Get best time of day (from behavioral features or model)
        best_hour = account_features.get('best_contact_time_of_day', 14)  # Default 2pm

        # Get best day of week (1=Monday, 7=Sunday)
        best_day = account_features.get('best_day_of_week', 3)  # Default Wednesday

        # Calculate next best date
        now = datetime.now()
        current_day = now.weekday() + 1

        days_until_best = (best_day - current_day) % 7
        if days_until_best == 0:
            days_until_best = 7  # Wait until next week

        next_contact_date = now + timedelta(days=days_until_best)

        # Set time
        next_contact_time = next_contact_date.replace(
            hour=best_hour,
            minute=0,
            second=0,
            microsecond=0
        )

        # Ensure within allowed hours
        min_hour = self.constraints['regulatory']['no_contact_before_hour']
        max_hour = self.constraints['regulatory']['no_contact_after_hour']

        if next_contact_time.hour < min_hour:
            next_contact_time = next_contact_time.replace(hour=min_hour)
        elif next_contact_time.hour >= max_hour:
            next_contact_time = next_contact_time.replace(hour=max_hour - 1)

        return next_contact_time

    def _calculate_confidence(
        self,
        channel: Dict,
        model_scores: Dict
    ) -> float:
        """Calculate confidence in recommendation"""

        # Base confidence from model
        if 'channel_probabilities' in model_scores:
            return channel.get('expected_response_rate', 0.5)
        else:
            return 0.5  # Medium confidence for rule-based

    def _evaluate_rule(
        self,
        rule_condition: str,
        account_features: Dict,
        persona: str
    ) -> bool:
        """Evaluate a rule condition"""

        # Simple rule evaluation
        # In production, use a proper rule engine

        if "cease_and_desist_flag = true" in rule_condition:
            return account_features.get('cease_and_desist_flag', False)

        if "phone_disconnected = true" in rule_condition:
            return account_features.get('phone_disconnected', False)

        if "segment =" in rule_condition:
            target_segment = rule_condition.split("=")[1].strip().strip('"')
            return persona == target_segment

        return False

    def _historical_best_channel(
        self,
        contact_history: pd.DataFrame
    ) -> Dict:
        """Fallback: Find historically best channel"""

        if len(contact_history) == 0:
            return {
                'channel': 'sms',
                'reason': 'Default (no history)',
                'expected_response_rate': 0.3
            }

        # Calculate response rate by channel
        channel_performance = contact_history.groupby('contact_channel').agg({
            'contact_outcome': lambda x: (x == 'payment').sum() / len(x)
        }).reset_index()

        channel_performance.columns = ['channel', 'response_rate']

        best = channel_performance.sort_values('response_rate', ascending=False).iloc[0]

        return {
            'channel': best['channel'],
            'reason': f'Historical best (response rate: {best["response_rate"]:.2f})',
            'expected_response_rate': best['response_rate']
        }


class NextBestOfferEngine:
    """
    Determines optimal collection offer for each account:
    - Settlement vs Payment Plan vs Legal Action vs Debt Sale
    - Discount percentage (if settlement)
    - Payment plan terms (if payment plan)
    - Expected recovery and ROI
    """

    def __init__(self, config: Dict):
        self.config = config
        self.offer_strategy = config['next_best_action']['offer_strategy']
        self.constraints = config['constraints']

    def recommend_offer(
        self,
        account_features: Dict,
        persona: str,
        model_scores: Dict,
        historical_offers: pd.DataFrame
    ) -> Dict:
        """
        Recommend next best offer for an account.

        Returns:
            {
                'offer_type': 'settlement',
                'settlement_discount_pct': 50,
                'settlement_amount': 2500,
                'payment_plan_months': None,
                'expected_recovery': 2400,
                'expected_acceptance_probability': 0.40,
                'expected_roi': 8.0,
                'rationale': 'High LTP, low ATP - settlement recommended'
            }
        """

        # Extract scores
        ltp_score = model_scores.get('ltp_score', 0.5)
        atp_score = model_scores.get('atp_score', 0.5)
        current_balance = account_features['current_balance']
        days_past_due = account_features['days_past_due']

        # Evaluate decision tree
        decision = self._evaluate_offer_decision_tree(
            ltp_score, atp_score, current_balance, days_past_due, persona
        )

        # Build offer details
        if decision['offer_type'] == 'settlement':
            offer = self._build_settlement_offer(
                current_balance, ltp_score, atp_score, decision.get('params', {})
            )

        elif decision['offer_type'] == 'payment_plan':
            offer = self._build_payment_plan_offer(
                current_balance, ltp_score, atp_score, decision.get('params', {})
            )

        elif decision['offer_type'] == 'legal_action':
            offer = self._build_legal_action(
                current_balance, decision.get('params', {})
            )

        elif decision['offer_type'] == 'debt_sale':
            offer = self._build_debt_sale(
                current_balance, decision.get('params', {})
            )

        else:
            offer = self._build_standard_contact(current_balance)

        # Add expected outcomes
        offer['expected_acceptance_probability'] = self._estimate_acceptance(
            offer, model_scores, historical_offers
        )

        offer['expected_recovery'] = (
            offer.get('settlement_amount', current_balance) *
            offer['expected_acceptance_probability']
        )

        offer['expected_cost'] = self._estimate_offer_cost(offer['offer_type'])

        offer['expected_roi'] = (
            (offer['expected_recovery'] - offer['expected_cost']) /
            offer['expected_cost'] if offer['expected_cost'] > 0 else 0
        )

        return offer

    def _evaluate_offer_decision_tree(
        self,
        ltp_score: float,
        atp_score: float,
        current_balance: float,
        days_past_due: int,
        persona: str
    ) -> Dict:
        """Evaluate offer decision tree from config"""

        for rule in self.offer_strategy['decision_tree']:
            condition = rule['if']

            # Evaluate condition
            if self._check_condition(
                condition, ltp_score, atp_score, current_balance,
                days_past_due, persona
            ):
                return {
                    'offer_type': rule['then'].replace('offer_', '').replace('recommend_', ''),
                    'params': rule.get('params', {})
                }

        # Default
        return {
            'offer_type': 'standard_contact',
            'params': {'offer_type': 'payment_reminder'}
        }

    def _check_condition(
        self,
        condition: str,
        ltp_score: float,
        atp_score: float,
        current_balance: float,
        days_past_due: int,
        persona: str
    ) -> bool:
        """Check if condition is true"""

        # Simple condition evaluation
        condition = condition.lower()

        if "ltp_score <" in condition:
            threshold = float(condition.split("<")[1].split("and")[0].strip())
            if ltp_score >= threshold:
                return False

        if "ltp_score >" in condition:
            threshold = float(condition.split(">")[1].split("and")[0].strip())
            if ltp_score <= threshold:
                return False

        if "atp_score <" in condition:
            threshold = float(condition.split("atp_score <")[1].split("and")[0].strip())
            if atp_score >= threshold:
                return False

        if "atp_score >" in condition:
            threshold = float(condition.split("atp_score >")[1].split("and")[0].strip())
            if atp_score <= threshold:
                return False

        if "current_balance <" in condition:
            threshold = float(condition.split("current_balance <")[1].split("and")[0].strip())
            if current_balance >= threshold:
                return False

        if "days_past_due >" in condition:
            threshold = int(condition.split("days_past_due >")[1].split("and")[0].strip())
            if days_past_due <= threshold:
                return False

        if "segment =" in condition:
            target_segment = condition.split("segment =")[1].split("and")[0].strip().strip('"')
            if persona != target_segment:
                return False

        return True

    def _build_settlement_offer(
        self,
        current_balance: float,
        ltp_score: float,
        atp_score: float,
        params: Dict
    ) -> Dict:
        """Build settlement offer details"""

        # Determine discount percentage
        discount_range = params.get('discount_range', '40-60%')
        min_discount, max_discount = [
            int(x.replace('%', '')) for x in discount_range.split('-')
        ]

        # Higher discount for lower LTP/ATP
        score_avg = (ltp_score + atp_score) / 2
        discount_pct = max_discount - (score_avg * (max_discount - min_discount))

        settlement_amount = current_balance * (1 - discount_pct / 100)

        return {
            'offer_type': 'settlement',
            'settlement_discount_pct': round(discount_pct, 1),
            'settlement_amount': round(settlement_amount, 2),
            'payment_plan_months': None,
            'payment_plan_monthly': None,
            'urgency': params.get('urgency', 'medium'),
            'rationale': f'Settlement with {discount_pct:.0f}% discount (LTP:{ltp_score:.2f}, ATP:{atp_score:.2f})'
        }

    def _build_payment_plan_offer(
        self,
        current_balance: float,
        ltp_score: float,
        atp_score: float,
        params: Dict
    ) -> Dict:
        """Build payment plan offer details"""

        # Determine plan length
        plan_months_range = params.get('plan_months', '6-12')
        min_months, max_months = [int(x) for x in plan_months_range.split('-')]

        # Longer plan for lower ATP
        plan_months = int(min_months + (1 - atp_score) * (max_months - min_months))

        # Small discount for payment plan
        discount_range = params.get('discount', '0-10%')
        max_discount = int(discount_range.split('-')[1].replace('%', ''))
        discount_pct = (1 - ltp_score) * max_discount

        adjusted_balance = current_balance * (1 - discount_pct / 100)
        monthly_payment = adjusted_balance / plan_months

        return {
            'offer_type': 'payment_plan',
            'settlement_discount_pct': round(discount_pct, 1),
            'settlement_amount': None,
            'payment_plan_months': plan_months,
            'payment_plan_monthly': round(monthly_payment, 2),
            'rationale': f'{plan_months}-month plan with {discount_pct:.0f}% discount'
        }

    def _build_legal_action(self, current_balance: float, params: Dict) -> Dict:
        """Build legal action recommendation"""
        return {
            'offer_type': 'legal_action',
            'settlement_discount_pct': 0,
            'settlement_amount': current_balance,
            'payment_plan_months': None,
            'payment_plan_monthly': None,
            'rationale': 'Low response rate - legal action recommended'
        }

    def _build_debt_sale(self, current_balance: float, params: Dict) -> Dict:
        """Build debt sale recommendation"""
        sale_value = current_balance * 0.10  # Typical 10 cents on dollar

        return {
            'offer_type': 'debt_sale',
            'settlement_discount_pct': 90,
            'settlement_amount': sale_value,
            'payment_plan_months': None,
            'payment_plan_monthly': None,
            'rationale': 'Low recovery probability - sell to agency'
        }

    def _build_standard_contact(self, current_balance: float) -> Dict:
        """Build standard contact (no special offer)"""
        return {
            'offer_type': 'payment_reminder',
            'settlement_discount_pct': 0,
            'settlement_amount': current_balance,
            'payment_plan_months': None,
            'payment_plan_monthly': None,
            'rationale': 'Standard payment reminder'
        }

    def _estimate_acceptance(
        self,
        offer: Dict,
        model_scores: Dict,
        historical_offers: pd.DataFrame
    ) -> float:
        """Estimate probability of offer acceptance"""

        # Use model scores if available
        offer_scores = model_scores.get('offer_scores', {})

        if offer['offer_type'] == 'settlement':
            return offer_scores.get('settlement_acceptance', 0.30)
        elif offer['offer_type'] == 'payment_plan':
            return offer_scores.get('payment_plan_acceptance', 0.40)
        else:
            return 0.20

    def _estimate_offer_cost(self, offer_type: str) -> float:
        """Estimate cost of making offer"""

        cost_map = {
            'settlement': 25.0,  # Processing + communication
            'payment_plan': 30.0,  # Higher admin cost
            'legal_action': 200.0,  # Attorney fees
            'debt_sale': 5.0,  # Low cost
            'payment_reminder': 2.0  # Low cost
        }

        return cost_map.get(offer_type, 10.0)


# Example usage
if __name__ == "__main__":
    import yaml

    # Load config
    with open("conf/use_cases/debt_collection_optimization.yaml") as f:
        config = yaml.safe_load(f)

    # Initialize engines
    action_engine = NextBestActionEngine(config)
    offer_engine = NextBestOfferEngine(config)

    # Example account
    account_features = {
        'account_id': 'ACC123',
        'current_balance': 5000,
        'days_past_due': 120,
        'phone_disconnected': False,
        'cease_and_desist_flag': False,
        'best_contact_time_of_day': 14,
        'best_day_of_week': 3
    }

    model_scores = {
        'ltp_score': 0.45,
        'atp_score': 0.60,
        'channel_probabilities': {
            'email': 0.15,
            'sms': 0.55,
            'call': 0.30
        }
    }

    contact_history = pd.DataFrame()

    # Get recommendations
    action = action_engine.recommend_action(
        account_features, 'high_value_cooperative', model_scores, contact_history
    )

    offer = offer_engine.recommend_offer(
        account_features, 'high_value_cooperative', model_scores, pd.DataFrame()
    )

    print("\n=== NEXT BEST ACTION ===")
    print(f"Channel: {action['recommended_channel']}")
    print(f"Time: {action['recommended_time']}")
    print(f"Confidence: {action['confidence']:.2f}")
    print(f"Reason: {action['reason']}")

    print("\n=== NEXT BEST OFFER ===")
    print(f"Offer Type: {offer['offer_type']}")
    print(f"Settlement Amount: ${offer.get('settlement_amount', 0):,.2f}")
    print(f"Discount: {offer['settlement_discount_pct']:.1f}%")
    print(f"Expected Recovery: ${offer['expected_recovery']:,.2f}")
    print(f"Expected ROI: {offer['expected_roi']:.1f}x")
    print(f"Rationale: {offer['rationale']}")
