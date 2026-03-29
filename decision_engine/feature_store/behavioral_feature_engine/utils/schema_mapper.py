"""
Interactive Schema Mapper - Maps BFE requirements to client data schemas

This utility allows the BFE repository to discover and map client data fields
at runtime without requiring upfront schema knowledge.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class SchemaMapper:
    """
    Interactive schema mapper that prompts for field mappings on first use
    and persists them for future executions.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to save/load schema mapping config
        """
        self.config_path = config_path or Path.home() / ".bfe_schema_mapping.json"
        self.mappings: Dict[str, Dict[str, str]] = {}
        self.load_mappings()

    def load_mappings(self):
        """Load existing schema mappings if available"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                self.mappings = json.load(f)
            print(f"✓ Loaded schema mappings from {self.config_path}")

    def save_mappings(self):
        """Persist schema mappings to disk"""
        with open(self.config_path, 'w') as f:
            json.dump(self.mappings, f, indent=2)
        print(f"✓ Schema mappings saved to {self.config_path}")

    def get_field_mapping(
        self,
        domain: str,
        required_fields: List[Dict[str, Any]],
        optional_fields: List[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Get or create field mappings for a domain (e.g., 'delinquency', 'payment')

        Args:
            domain: Feature domain name
            required_fields: List of required field definitions
                [{"bfe_name": "dpd", "description": "Days Past Due", "type": "int"}]
            optional_fields: List of optional field definitions

        Returns:
            Dictionary mapping BFE field names to client field names
        """
        optional_fields = optional_fields or []

        # Check if mappings already exist
        if domain in self.mappings:
            print(f"\n✓ Using existing schema mappings for '{domain}' domain")
            return self.mappings[domain]

        # Interactive mapping
        print(f"\n{'='*70}")
        print(f"SCHEMA MAPPING REQUIRED: {domain.upper()} Domain")
        print(f"{'='*70}")
        print(f"\nThe BFE repository needs to map its field requirements to your data schema.")
        print(f"Please provide the column names from your data that correspond to each BFE field.\n")

        mappings = {}

        # Map required fields
        print("REQUIRED FIELDS:")
        print("-" * 70)
        for field_def in required_fields:
            bfe_name = field_def["bfe_name"]
            description = field_def["description"]
            field_type = field_def.get("type", "any")
            example = field_def.get("example", "")

            print(f"\n  BFE Field: {bfe_name}")
            print(f"  Description: {description}")
            print(f"  Type: {field_type}")
            if example:
                print(f"  Example: {example}")

            client_field = input(f"  → Your column name: ").strip()

            if not client_field:
                raise ValueError(f"Required field '{bfe_name}' must be mapped!")

            mappings[bfe_name] = client_field
            print(f"  ✓ Mapped: {bfe_name} → {client_field}")

        # Map optional fields
        if optional_fields:
            print(f"\n{'='*70}")
            print("OPTIONAL FIELDS (press Enter to skip):")
            print("-" * 70)
            for field_def in optional_fields:
                bfe_name = field_def["bfe_name"]
                description = field_def["description"]
                field_type = field_def.get("type", "any")

                print(f"\n  BFE Field: {bfe_name}")
                print(f"  Description: {description}")
                print(f"  Type: {field_type}")

                client_field = input(f"  → Your column name (or Enter to skip): ").strip()

                if client_field:
                    mappings[bfe_name] = client_field
                    print(f"  ✓ Mapped: {bfe_name} → {client_field}")
                else:
                    print(f"  ⊗ Skipped: {bfe_name}")

        # Save mappings
        self.mappings[domain] = mappings
        self.save_mappings()

        print(f"\n{'='*70}")
        print(f"✓ Schema mapping complete for '{domain}' domain")
        print(f"{'='*70}\n")

        return mappings

    def get_mapped_field(self, domain: str, bfe_field: str) -> Optional[str]:
        """Get client field name for a BFE field"""
        return self.mappings.get(domain, {}).get(bfe_field)

    def list_mappings(self) -> Dict[str, Dict[str, str]]:
        """Return all current mappings"""
        return self.mappings

    def reset_domain(self, domain: str):
        """Clear mappings for a specific domain"""
        if domain in self.mappings:
            del self.mappings[domain]
            self.save_mappings()
            print(f"✓ Reset schema mappings for '{domain}' domain")

    def reset_all(self):
        """Clear all mappings"""
        self.mappings = {}
        self.save_mappings()
        print("✓ Reset all schema mappings")


# Singleton instance
_schema_mapper = None

def get_schema_mapper(config_path: Optional[str] = None) -> SchemaMapper:
    """Get or create global schema mapper instance"""
    global _schema_mapper
    if _schema_mapper is None:
        _schema_mapper = SchemaMapper(config_path)
    return _schema_mapper
