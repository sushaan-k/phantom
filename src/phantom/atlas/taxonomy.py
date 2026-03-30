"""MITRE ATLAS technique definitions and taxonomy loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from phantom.exceptions import TaxonomyError
from phantom.logging import get_logger
from phantom.models import Severity

logger = get_logger("phantom.atlas.taxonomy")

_DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "techniques.json"


class SubTechnique(BaseModel):
    """A sub-technique within a MITRE ATLAS technique."""

    id: str
    name: str
    description: str


class Technique(BaseModel):
    """A MITRE ATLAS technique with its associated metadata."""

    id: str
    name: str
    tactic: str
    description: str
    subtechniques: list[SubTechnique] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)
    severity_default: Severity = Severity.MEDIUM
    references: list[str] = Field(default_factory=list)

    def get_subtechnique(self, sub_id: str) -> SubTechnique | None:
        """Look up a sub-technique by ID.

        Args:
            sub_id: The sub-technique ID (e.g., 'AML.T0051.001').

        Returns:
            The matching SubTechnique, or None.
        """
        for sub in self.subtechniques:
            if sub.id == sub_id:
                return sub
        return None


class CategoryMapping(BaseModel):
    """Mapping from an attack category to ATLAS technique IDs."""

    techniques: list[str]
    description: str


class ATLASTaxonomy:
    """Loader and query interface for the MITRE ATLAS technique database.

    Loads technique definitions from a JSON file and provides
    lookup methods used by the ATLASMapper.

    Args:
        data_path: Path to the ATLAS techniques JSON file. Defaults
            to the bundled atlas_data/techniques.json.
    """

    def __init__(self, data_path: Path | str | None = None) -> None:
        self._data_path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
        self._techniques: dict[str, Technique] = {}
        self._category_map: dict[str, CategoryMapping] = {}
        self._loaded = False

    def load(self) -> None:
        """Load the ATLAS technique database from disk.

        Raises:
            TaxonomyError: If the data file is missing or malformed.
        """
        if not self._data_path.exists():
            raise TaxonomyError(f"Technique database not found at {self._data_path}")

        try:
            raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise TaxonomyError(f"Failed to parse technique database: {exc}") from exc

        self._parse_techniques(raw)
        self._parse_categories(raw)
        self._loaded = True

        logger.info(
            "taxonomy_loaded",
            technique_count=len(self._techniques),
            category_count=len(self._category_map),
        )

    def _parse_techniques(self, raw: dict[str, Any]) -> None:
        """Parse technique entries from raw JSON data.

        Args:
            raw: The parsed JSON dictionary.
        """
        for entry in raw.get("techniques", []):
            technique = Technique(
                id=entry["id"],
                name=entry["name"],
                tactic=entry["tactic"],
                description=entry["description"],
                subtechniques=[
                    SubTechnique(**sub) for sub in entry.get("subtechniques", [])
                ],
                mitigations=entry.get("mitigations", []),
                severity_default=Severity(entry.get("severity_default", "MEDIUM")),
                references=entry.get("references", []),
            )
            self._techniques[technique.id] = technique

            for sub in technique.subtechniques:
                self._techniques[sub.id] = technique

    def _parse_categories(self, raw: dict[str, Any]) -> None:
        """Parse attack category mappings from raw JSON data.

        Args:
            raw: The parsed JSON dictionary.
        """
        for cat_name, cat_data in raw.get("attack_categories", {}).items():
            self._category_map[cat_name] = CategoryMapping(
                techniques=cat_data["techniques"],
                description=cat_data["description"],
            )

    def _ensure_loaded(self) -> None:
        """Load the taxonomy if it hasn't been loaded yet."""
        if not self._loaded:
            self.load()

    def get_technique(self, technique_id: str) -> Technique | None:
        """Look up a technique by its ID.

        Args:
            technique_id: The ATLAS technique ID (e.g., 'AML.T0051').

        Returns:
            The Technique if found, else None.
        """
        self._ensure_loaded()
        return self._techniques.get(technique_id)

    def get_techniques_for_category(self, category: str) -> list[Technique]:
        """Get all techniques associated with an attack category.

        Args:
            category: The attack category name (e.g., 'prompt_injection').

        Returns:
            A list of Technique objects.
        """
        self._ensure_loaded()
        mapping = self._category_map.get(category)
        if not mapping:
            return []

        seen: set[str] = set()
        results: list[Technique] = []
        for tid in mapping.techniques:
            tech = self._techniques.get(tid)
            if tech and tech.id not in seen:
                seen.add(tech.id)
                results.append(tech)
        return results

    def get_default_severity(self, technique_id: str) -> Severity:
        """Get the default severity for a technique.

        Args:
            technique_id: The ATLAS technique ID.

        Returns:
            The default Severity, or MEDIUM if the technique is unknown.
        """
        self._ensure_loaded()
        technique = self._techniques.get(technique_id)
        if technique:
            return technique.severity_default
        return Severity.MEDIUM

    def get_mitigations(self, technique_id: str) -> list[str]:
        """Get recommended mitigations for a technique.

        Args:
            technique_id: The ATLAS technique ID.

        Returns:
            A list of mitigation strings.
        """
        self._ensure_loaded()
        technique = self._techniques.get(technique_id)
        if technique:
            return technique.mitigations
        return []

    @property
    def all_technique_ids(self) -> list[str]:
        """Return all loaded technique IDs."""
        self._ensure_loaded()
        return list(self._techniques.keys())

    @property
    def all_categories(self) -> list[str]:
        """Return all loaded attack category names."""
        self._ensure_loaded()
        return list(self._category_map.keys())
