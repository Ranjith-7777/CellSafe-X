"""Tests for the Phase-7B canonical sensor registry
(config.model_parameters.SENSOR_REGISTRY / OBS_CHANNEL_TO_SENSOR_FAMILY)."""

from __future__ import annotations

from config import model_parameters as P


def test_registry_covers_every_overlapping_sensor_name():
    for name in P.OVERLAPPING_SENSOR_NAMES:
        assert f"temp_{name}" in P.SENSOR_REGISTRY


def test_overlapping_family_members_are_flagged_overlaps_true():
    core = [k for k, v in P.SENSOR_REGISTRY.items() if v["family"] == "cell_core_temperature"]
    assert len(core) == len(P.OVERLAPPING_SENSOR_NAMES)
    assert all(P.SENSOR_REGISTRY[k]["overlaps"] for k in core)


def test_non_overlapping_sensors_are_flagged_overlaps_false():
    singleton_families = ["voltage", "gas", "current", "state_of_charge", "cooling"]
    for k, v in P.SENSOR_REGISTRY.items():
        if v["family"] in singleton_families:
            assert v["overlaps"] is False, k


def test_exactly_one_active_sensing_representative_per_overlapping_family():
    """Goal 4: EIG must never see primary/surface/backup as three
    independent active-sensing choices - only one representative per family."""
    core_active = [
        k for k, v in P.SENSOR_REGISTRY.items()
        if v["family"] == "cell_core_temperature" and v["active_sensing_eligible"]
    ]
    assert len(core_active) == 1
    assert core_active[0] == "temp_primary"


def test_hmm_eligible_sensors_are_marginalised_not_required():
    for k, v in P.SENSOR_REGISTRY.items():
        if v["hmm_eligible"]:
            assert v["missing_evidence_support"] == "marginalised", k


def test_contextual_only_sensors_are_required_not_marginalised():
    for k, v in P.SENSOR_REGISTRY.items():
        if v["evidence_type"] == "contextual":
            assert not v["hmm_eligible"]
            assert v["missing_evidence_support"] == "required", k


def test_obs_channel_family_map_covers_every_obs_channel():
    assert set(P.OBS_CHANNEL_TO_SENSOR_FAMILY) == set(P.OBS_CHANNELS)


def test_active_sensing_candidate_channels_all_have_a_registered_family():
    for ch in P.ACTIVE_SENSING_CANDIDATE_CHANNELS:
        family = P.OBS_CHANNEL_TO_SENSOR_FAMILY[ch]
        assert any(v["family"] == family for v in P.SENSOR_REGISTRY.values())


def test_temp_rate_shares_the_core_temperature_family_not_a_fake_sensor():
    assert P.OBS_CHANNEL_TO_SENSOR_FAMILY["temp_rate"] == "cell_core_temperature"
    assert "temp_rate" not in P.SENSOR_REGISTRY  # no invented raw sensor for a derived channel
