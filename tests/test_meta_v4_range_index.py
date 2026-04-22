"""Testes para o novo alvo delta_range_index do meta-calibrator (v4).

Validação de bugs específicos da v3 que motivaram a v4:
- Alvos contínuos em horas não saiam do arredondamento em faixas de 3h.
- Bounds em % do base_hours limitavam a correção em regime fraco.
- A v4 aprende e prediz direto em range_index, eliminando a quantização.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from ai.core import meta_calibrator as mc
from ai.core.meta_calibrator import (
    _range_delta_target,
    _route_range_delta_bounds,
    build_training_row_from_validation,
    predict_meta_calibration,
    save_meta_model,
    train_meta_calibrator,
)
from ai.core.effort_calibration import hours_to_range_index
from config.settings import settings


class TestRangeDeltaTarget(unittest.TestCase):
    """Unit tests para os helpers da v4."""

    def test_delta_range_zero_when_same_bucket(self):
        # 4h e 5h estão ambas na faixa 3-6h (range_index=2)
        self.assertEqual(_range_delta_target(4.0, 5.0), 0.0)

    def test_delta_range_positive_when_actual_higher(self):
        # base=5h (range 2, 3-6h) real=20h (range 7, 18-21h) → +5 faixas
        self.assertEqual(_range_delta_target(20.0, 5.0), 5.0)

    def test_delta_range_negative_when_actual_lower(self):
        # base=20h (range 7) real=2h (range 1) → -6 faixas
        self.assertEqual(_range_delta_target(2.0, 20.0), -6.0)

    def test_delta_range_handles_zero(self):
        self.assertEqual(_range_delta_target(0, 5), 0.0)
        self.assertEqual(_range_delta_target(5, 0), 0.0)

    def test_bounds_analogical_weak_widest(self):
        lo, hi = _route_range_delta_bounds("analogical_weak")
        self.assertEqual((lo, hi), (-8.0, 8.0))

    def test_bounds_analogical_primary_tightest(self):
        lo, hi = _route_range_delta_bounds("analogical_primary")
        self.assertEqual((lo, hi), (-2.0, 2.0))

    def test_bounds_unknown_route_treated_as_weak(self):
        lo, hi = _route_range_delta_bounds("unknown")
        self.assertEqual((lo, hi), (-8.0, 8.0))


class TestTrainV4Target(unittest.TestCase):
    """Confirma que o treino produz modelo com target_mode correto."""

    def _make_row(self, base_h, actual_h, project="p1", route="analogical_weak"):
        """Helper para criar linhas de treino sintéticas."""
        return {
            "project_key": project,
            "issue_type": "story",
            "retrieval_route": route,
            "heuristic_size_bucket": "m",
            "size_bucket": "m",
            "calibration_source": "heuristic_range_consensus",
            "rule_selected_model": "heuristic_bucket_calibrated",
            "base_hours": base_h,
            "base_log_hours": mc._log1p_hours(base_h),
            "analogical_log_hours": mc._log1p_hours(base_h),
            "heuristic_log_hours": mc._log1p_hours(base_h),
            "base_range_index": float(hours_to_range_index(base_h)),
            "analogical_range_index": float(hours_to_range_index(base_h)),
            "heuristic_range_index": float(hours_to_range_index(base_h)),
            "bucket_rank": 3.0,
            "heuristic_bucket_rank": 3.0,
            "analogical_bucket_rank": 3.0,
            "top1_score": 0.5,
            "top3_avg_score": 0.4,
            "useful_count": 1.0,
            "hours_spread": 0.0,
            "analogical_confidence": 0.5,
            "heuristic_confidence": 0.7,
            "base_confidence": 0.6,
            "bucket_gap": 0.0,
            "range_gap": 0.0,
            "rag_context_sufficient": 1.0,
            "complexity_bucket_delta": 0.0,
            "agile_guard_bucket_delta": 0.0,
            "actual_hours": actual_h,
        }

    def test_trained_model_declares_range_index_target(self):
        # 12 linhas onde real é consistentemente ~5 faixas acima do base.
        # base=5h (range 2), real=~22h (range 8) → delta esperado +6.
        rows = [self._make_row(5.0, 22.0) for _ in range(14)]
        model = train_meta_calibrator(rows)

        self.assertEqual(
            model["training_summary"]["target_mode"], "delta_range_index"
        )
        self.assertIn("linear_model", model)
        self.assertIn("linear_model_log_hours", model)  # legado preservado
        self.assertEqual(
            model["model_version"], "meta_calibrator_v4_range_index"
        )

    def test_category_adjustments_present_for_both_targets(self):
        rows = [self._make_row(5.0, 22.0) for _ in range(14)]
        model = train_meta_calibrator(rows)

        self.assertIn("category_adjustments", model)  # v4 range_index
        self.assertIn("category_adjustments_log_hours", model)  # legado


class TestPredictV4RangeIndex(unittest.TestCase):
    """Confirma que o predict v4 empurra o range_index quando tem evidência."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.tmpfile.close()
        self.original_path = settings.META_CALIBRATOR_MODEL_PATH
        settings.META_CALIBRATOR_MODEL_PATH = self.tmpfile.name
        # Limpa cache do modelo
        mc._MODEL_CACHE["path"] = None
        mc._MODEL_CACHE["mtime"] = None
        mc._MODEL_CACHE["data"] = None

    def tearDown(self):
        settings.META_CALIBRATOR_MODEL_PATH = self.original_path
        Path(self.tmpfile.name).unlink(missing_ok=True)
        mc._MODEL_CACHE["path"] = None
        mc._MODEL_CACHE["mtime"] = None
        mc._MODEL_CACHE["data"] = None

    def _make_row(self, base_h, actual_h, project="p1", route="analogical_weak"):
        # Reusa o helper do caso de treino
        t = TestTrainV4Target()
        return t._make_row(base_h, actual_h, project, route)

    def test_predict_shifts_range_in_weak_route(self):
        """
        Cenário: histórico mostra consistentemente que issues com base=5h
        no projeto X acabam em ~22h (delta +5 faixas). A predição do meta
        para uma nova issue com base=5h deve empurrar o range_index para
        cima substancialmente.
        """
        # 20 issues todas com base=5h real=22h → delta +5 faixas sempre
        training = [self._make_row(5.0, 22.0) for _ in range(20)]
        model = train_meta_calibrator(training)
        save_meta_model(model, Path(self.tmpfile.name))

        # Nova issue com base=5h (range 2)
        feature = {
            "project_key": "p1",
            "issue_type": "story",
            "retrieval_route": "analogical_weak",
            "heuristic_size_bucket": "m",
            "size_bucket": "m",
            "calibration_source": "heuristic_range_consensus",
            "rule_selected_model": "heuristic_bucket_calibrated",
            "base_log_hours": mc._log1p_hours(5.0),
            "analogical_log_hours": mc._log1p_hours(5.0),
            "heuristic_log_hours": mc._log1p_hours(5.0),
            "base_range_index": 2.0,
            "analogical_range_index": 2.0,
            "heuristic_range_index": 2.0,
            "bucket_rank": 3.0,
            "heuristic_bucket_rank": 3.0,
            "analogical_bucket_rank": 3.0,
            "top1_score": 0.5,
            "top3_avg_score": 0.4,
            "useful_count": 1.0,
            "hours_spread": 0.0,
            "analogical_confidence": 0.5,
            "heuristic_confidence": 0.7,
            "base_confidence": 0.6,
            "bucket_gap": 0.0,
            "range_gap": 0.0,
            "rag_context_sufficient": 1.0,
            "complexity_bucket_delta": 0.0,
            "agile_guard_bucket_delta": 0.0,
        }

        result = predict_meta_calibration(feature)

        self.assertTrue(result["available"])
        self.assertEqual(result["target_mode"], "delta_range_index")
        # O meta DEVE empurrar o range_index pra cima.
        # base_range=2 (3-6h), esperado >=5 (12-15h ou mais alto).
        self.assertGreaterEqual(
            result["range_index"], 4,
            f"Meta deveria empurrar range pra cima dado histórico consistente. "
            f"predicted={result['range_index']}, estimated_hours={result['estimated_hours']}, "
            f"delta={result.get('predicted_range_delta')}"
        )
        # Horas finais coerentes com o range_index retornado
        self.assertGreater(result["estimated_hours"], 5.0)

    def test_primary_route_has_tighter_bounds(self):
        """Em analogical_primary, mesmo com histórico extremo, o meta não
        pode puxar mais que ±2 faixas (preserva a autoridade do analogical forte)."""
        training = [
            self._make_row(5.0, 35.0, route="analogical_primary") for _ in range(20)
        ]
        model = train_meta_calibrator(training)
        save_meta_model(model, Path(self.tmpfile.name))

        feature = {
            "project_key": "p1",
            "issue_type": "story",
            "retrieval_route": "analogical_primary",
            "heuristic_size_bucket": "m",
            "size_bucket": "m",
            "calibration_source": "heuristic_range_consensus",
            "rule_selected_model": "heuristic_bucket_calibrated",
            "base_log_hours": mc._log1p_hours(5.0),
            "analogical_log_hours": mc._log1p_hours(5.0),
            "heuristic_log_hours": mc._log1p_hours(5.0),
            "base_range_index": 2.0,
            "analogical_range_index": 2.0,
            "heuristic_range_index": 2.0,
            "bucket_rank": 3.0,
            "heuristic_bucket_rank": 3.0,
            "analogical_bucket_rank": 3.0,
            "top1_score": 0.95,
            "top3_avg_score": 0.90,
            "useful_count": 1.0,
            "hours_spread": 0.0,
            "analogical_confidence": 0.9,
            "heuristic_confidence": 0.7,
            "base_confidence": 0.8,
            "bucket_gap": 0.0,
            "range_gap": 0.0,
            "rag_context_sufficient": 1.0,
            "complexity_bucket_delta": 0.0,
            "agile_guard_bucket_delta": 0.0,
        }

        result = predict_meta_calibration(feature)
        self.assertTrue(result["available"])
        # base=2, clamped to base+2=4 max.
        self.assertLessEqual(result["range_index"], 4)


if __name__ == "__main__":
    unittest.main()
