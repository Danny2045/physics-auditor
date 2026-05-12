"""Tests for ``physics_auditor.utils.pdb_provenance.verify_pdb_source``.

The helper is the gate that every PDB ingestion runner must pass through
before parsing. The tests below exercise:

1. Real-file match — 1D3H reports HOMO SAPIENS, 4ORM reports
   PLASMODIUM FALCIPARUM. Verification with the matching organism must
   return None silently.
2. Real-file mismatch — 1D3H verified against PLASMODIUM FALCIPARUM
   must raise with both the actual organism and the expected organism
   named in the message.
3. Missing file — non-existent path must raise with a message naming
   the file.
4. Malformed SOURCE — a synthetic PDB file with no ORGANISM_SCIENTIFIC
   field, and one with no SOURCE records at all, must each raise with
   informative messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_auditor.utils.pdb_provenance import verify_pdb_source

REPO_ROOT = Path(__file__).resolve().parent.parent
PDB_1D3H = REPO_ROOT / "benchmark" / "structures" / "experimental" / "1D3H.pdb"
PDB_4ORM = REPO_ROOT / "benchmark" / "structures" / "experimental" / "4ORM.pdb"


class TestRealFileMatch:
    def test_1d3h_homo_sapiens_passes(self):
        # Returns None on success; absence of an exception is the check.
        assert verify_pdb_source(PDB_1D3H, "HOMO SAPIENS") is None

    def test_4orm_plasmodium_falciparum_passes(self):
        assert verify_pdb_source(PDB_4ORM, "PLASMODIUM FALCIPARUM") is None

    def test_match_is_case_insensitive(self):
        assert verify_pdb_source(PDB_1D3H, "homo sapiens") is None
        assert verify_pdb_source(PDB_4ORM, "Plasmodium Falciparum") is None


class TestRealFileMismatch:
    def test_1d3h_against_plasmodium_raises(self):
        with pytest.raises(ValueError) as exc:
            verify_pdb_source(PDB_1D3H, "PLASMODIUM FALCIPARUM")
        msg = str(exc.value)
        assert "HOMO SAPIENS" in msg.upper()
        assert "PLASMODIUM FALCIPARUM" in msg.upper()

    def test_4orm_against_homo_sapiens_raises(self):
        with pytest.raises(ValueError) as exc:
            verify_pdb_source(PDB_4ORM, "HOMO SAPIENS")
        msg = str(exc.value)
        assert "PLASMODIUM FALCIPARUM" in msg.upper()
        assert "HOMO SAPIENS" in msg.upper()


class TestMissingFile:
    def test_missing_file_raises_with_path(self, tmp_path):
        missing = tmp_path / "does_not_exist.pdb"
        with pytest.raises(ValueError) as exc:
            verify_pdb_source(missing, "HOMO SAPIENS")
        msg = str(exc.value)
        assert str(missing) in msg
        assert "not found" in msg.lower()


class TestMalformedSource:
    def test_no_source_records_raises(self, tmp_path):
        # A PDB file with HEADER but no SOURCE block.
        broken = tmp_path / "no_source.pdb"
        broken.write_text(
            "HEADER    BROKEN TEST FILE\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
            "END\n"
        )
        with pytest.raises(ValueError) as exc:
            verify_pdb_source(broken, "HOMO SAPIENS")
        msg = str(exc.value)
        assert "SOURCE" in msg
        assert str(broken) in msg

    def test_source_without_organism_scientific_raises(self, tmp_path):
        # SOURCE records present but no ORGANISM_SCIENTIFIC key.
        broken = tmp_path / "no_organism.pdb"
        broken.write_text(
            "HEADER    BROKEN TEST FILE\n"
            "SOURCE    MOL_ID: 1;\n"
            "SOURCE   2 EXPRESSION_SYSTEM: ESCHERICHIA COLI;\n"
            "END\n"
        )
        with pytest.raises(ValueError) as exc:
            verify_pdb_source(broken, "HOMO SAPIENS")
        msg = str(exc.value)
        assert "ORGANISM_SCIENTIFIC" in msg

    def test_empty_organism_scientific_value_raises(self, tmp_path):
        # SOURCE has the key but the value is empty.
        broken = tmp_path / "empty_organism.pdb"
        broken.write_text(
            "HEADER    BROKEN TEST FILE\n"
            "SOURCE    MOL_ID: 1;\n"
            "SOURCE   2 ORGANISM_SCIENTIFIC: ;\n"
            "END\n"
        )
        with pytest.raises(ValueError) as exc:
            verify_pdb_source(broken, "HOMO SAPIENS")
        msg = str(exc.value)
        assert "ORGANISM_SCIENTIFIC" in msg
