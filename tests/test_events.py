import unittest
import pandas as pd
from clinical_multimodal.evaluation.events import continuous_warning_metrics
from clinical_multimodal.profiling import memory_snapshot


class EventTests(unittest.TestCase):
    def setUp(self):
        self.episodes = pd.DataFrame([
            {"patient_id":"p1", "episode_id":"e1", "monitor_start":"2020-01-01T00:00:00Z",
             "monitor_end":"2020-01-02T00:00:00Z", "event_time":"2020-01-02T00:00:00Z"},
            {"patient_id":"p2", "episode_id":"e2", "monitor_start":"2020-01-01T00:00:00Z",
             "monitor_end":"2020-01-02T00:00:00Z", "event_time":None}])
        self.predictions = pd.DataFrame([
            {"patient_id":"p1", "episode_id":"e1", "prediction_time":"2020-01-01T20:00:00Z", "risk":.8},
            {"patient_id":"p1", "episode_id":"e1", "prediction_time":"2020-01-01T21:00:00Z", "risk":.8},
            {"patient_id":"p2", "episode_id":"e2", "prediction_time":"2020-01-01T10:00:00Z", "risk":.8},
            {"patient_id":"p2", "episode_id":"e2", "prediction_time":"2020-01-01T23:00:00Z", "risk":.8}])

    def test_lead_time_false_alarms_and_censoring(self):
        report = continuous_warning_metrics(self.predictions,self.episodes,horizon_hours=6,threshold=.5)
        self.assertEqual(report["event_detection_rate"],1)
        self.assertEqual(report["first_alarm_lead_hours"],[4])
        self.assertEqual(report["false_alarms"],1)
        self.assertEqual(report["unassessable_censored_alarms"],1)
        self.assertEqual(report["false_alarms_per_patient_day"],.5)

    def test_refractory_prevents_repeated_alarms(self):
        report = continuous_warning_metrics(self.predictions,self.episodes,horizon_hours=6,threshold=.5,refractory_hours=2)
        self.assertEqual(report["total_alarms_after_refractory"],3)

    def test_no_prediction_opportunity_not_silently_a_miss(self):
        report = continuous_warning_metrics(self.predictions.iloc[2:],self.episodes,horizon_hours=6,threshold=.5)
        self.assertIsNone(report["event_detection_rate"])
        self.assertEqual(report["events_without_prediction_opportunity"],1)

    def test_after_event_rejected(self):
        p = self.predictions.copy()
        p.loc[0,"prediction_time"] = "2020-01-02T01:00:00Z"
        with self.assertRaisesRegex(ValueError,"outside"):
            continuous_warning_metrics(p,self.episodes,horizon_hours=6,threshold=.5)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            continuous_warning_metrics(self.predictions,self.episodes,horizon_hours=float("nan"),threshold=.5)

    def test_memory_snapshot(self):
        snapshot = memory_snapshot()
        self.assertGreater(snapshot["process_lifetime_peak_rss_bytes"],0)
