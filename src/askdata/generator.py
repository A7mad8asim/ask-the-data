"""Synthetic data for a fictional network of 8 primary-care clinics in Doha.

No real patients: every row is drawn from seeded random distributions, so the same seed
always produces the same data. The patterns are built in on purpose, so analysis questions
have interesting answers and the README can document each one:

- a mostly expatriate, multinational population (South Asian workers cluster at the
  Industrial Area Clinic)
- high diabetes and vitamin D deficiency prevalence
- shorter clinic hours and more no-shows during Ramadan
- fewer visits in summer, when many residents travel
- small data-entry gaps (missing lab units, missing waiting times)

Run:  python -m askdata.generator            (60,000 patients, about 2.5 million rows)
"""

from __future__ import annotations

import argparse
import hashlib
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import Settings
from .schema import TABLES

START = date(2023, 1, 1)
END = date(2025, 12, 31)
N_DAYS = (END - START).days + 1

RAMADAN = [
    (date(2023, 3, 23), date(2023, 4, 20)),
    (date(2024, 3, 11), date(2024, 4, 9)),
    (date(2025, 3, 1), date(2025, 3, 29)),
]
EID = [  # Eid al-Fitr and Eid al-Adha holidays
    (date(2023, 4, 21), date(2023, 4, 23)),
    (date(2023, 6, 28), date(2023, 6, 30)),
    (date(2024, 4, 10), date(2024, 4, 12)),
    (date(2024, 6, 16), date(2024, 6, 18)),
    (date(2025, 3, 30), date(2025, 4, 1)),
    (date(2025, 6, 6), date(2025, 6, 8)),
]

# id, name, Arabic name, municipality, opens, closes, Ramadan opens, Ramadan closes, lab, mean wait (min)
CLINICS = [
    ("C01", "Doha Central Clinic", "عيادة الدوحة المركزية", "Doha", "07:00", "23:00", "09:00", "23:00", True, 26),
    ("C02", "West Bay Clinic", "عيادة الخليج الغربي", "Doha", "07:00", "21:00", "09:00", "22:00", True, 20),
    ("C03", "Al Rayyan Clinic", "عيادة الريان", "Al Rayyan", "07:00", "23:00", "09:00", "23:00", True, 29),
    ("C04", "Al Wakrah Clinic", "عيادة الوكرة", "Al Wakrah", "07:00", "21:00", "09:00", "22:00", True, 24),
    ("C05", "Umm Salal Clinic", "عيادة أم صلال", "Umm Salal", "07:00", "21:00", "09:00", "22:00", False, 19),
    ("C06", "Al Khor Clinic", "عيادة الخور", "Al Khor", "07:00", "19:00", "09:00", "21:00", False, 17),
    ("C07", "Industrial Area Clinic", "عيادة المنطقة الصناعية", "Al Rayyan", "06:00", "23:00", "08:00", "23:00", True, 44),
    ("C08", "Al Daayen Clinic", "عيادة الضعاين", "Al Daayen", "07:00", "21:00", "09:00", "22:00", False, 21),
]
CLINIC_IDS = np.array([c[0] for c in CLINICS])
CLINIC_MUNICIPALITY = np.array([c[3] for c in CLINICS])
CLINIC_WAIT = np.array([c[9] for c in CLINICS], dtype=float)
INDUSTRIAL = 6  # index of C07

NATIONALITIES = np.array(["Qatari", "Other Arab", "South Asian", "Southeast Asian", "African", "Western", "Other"])
NAT_P = [0.12, 0.18, 0.38, 0.14, 0.08, 0.05, 0.05]
QATARI, OTHER_ARAB, SOUTH_ASIAN, SE_ASIAN, AFRICAN, WESTERN, OTHER = range(7)

AGE_BANDS = np.array(["0-17", "18-29", "30-39", "40-49", "50-59", "60+"])
KID, A18, A30, A40, A50, A60 = range(6)

DEPARTMENTS = np.array(["General Practice", "Diabetes Clinic", "Women's Health", "Pediatrics", "Dental", "Vaccination"])
GP, DC, WH, PED, DEN, VAC = range(6)
SPECIALTY = np.array(["Family Medicine", "Endocrinology", "Obstetrics & Gynaecology", "Paediatrics", "Dentistry", "Nursing"])

CHANNELS = np.array(["App", "Phone", "Walk-in", "Referral"])
APP, PHONE, WALKIN, REFERRAL = range(4)

ICD10 = {  # code: (description, chapter, chronic)
    "E11.9": ("Type 2 diabetes mellitus without complications", "Endocrine", True),
    "E11.65": ("Type 2 diabetes mellitus with hyperglycemia", "Endocrine", True),
    "I10": ("Essential (primary) hypertension", "Circulatory", True),
    "E78.5": ("Hyperlipidemia, unspecified", "Endocrine", True),
    "E55.9": ("Vitamin D deficiency, unspecified", "Endocrine", True),
    "J45.909": ("Unspecified asthma, uncomplicated", "Respiratory", True),
    "J06.9": ("Acute upper respiratory infection, unspecified", "Respiratory", False),
    "J02.9": ("Acute pharyngitis, unspecified", "Respiratory", False),
    "A09": ("Infectious gastroenteritis and colitis, unspecified", "Infectious", False),
    "H66.90": ("Otitis media, unspecified", "Eye & ear", False),
    "H10.9": ("Unspecified conjunctivitis", "Eye & ear", False),
    "L30.9": ("Dermatitis, unspecified", "Skin", False),
    "R50.9": ("Fever, unspecified", "Symptoms & signs", False),
    "R51.9": ("Headache, unspecified", "Symptoms & signs", False),
    "M54.50": ("Low back pain, unspecified", "Musculoskeletal", False),
    "M79.10": ("Myalgia, unspecified site", "Musculoskeletal", False),
    "K29.70": ("Gastritis, unspecified, without bleeding", "Digestive", False),
    "B34.9": ("Viral infection, unspecified", "Infectious", False),
    "N39.0": ("Urinary tract infection, site not specified", "Genitourinary", False),
    "N94.6": ("Dysmenorrhea, unspecified", "Genitourinary", False),
    "Z34.90": ("Encounter for supervision of normal pregnancy", "Health services", False),
    "Z01.419": ("Encounter for gynecological examination", "Health services", False),
    "Z00.00": ("Encounter for general adult medical examination", "Health services", False),
    "Z00.129": ("Encounter for routine child health examination", "Health services", False),
    "Z23": ("Encounter for immunization", "Health services", False),
    "K02.9": ("Dental caries, unspecified", "Digestive", False),
    "K05.10": ("Chronic gingivitis, plaque induced", "Digestive", False),
    "K08.89": ("Other specified disorders of teeth and supporting structures", "Digestive", False),
}

CHRONIC_DRUGS = {
    "Biguanides (metformin)", "DPP-4 inhibitors", "SGLT2 inhibitors", "Insulin", "ACE inhibitors",
    "Calcium channel blockers", "Statins", "Vitamin D supplements", "Inhaled corticosteroids",
}
# diagnosis code -> [(drug class, probability)]
PRESCRIBING = {
    "E11.9": [("Biguanides (metformin)", 0.85), ("DPP-4 inhibitors", 0.30), ("SGLT2 inhibitors", 0.15)],
    "E11.65": [("Biguanides (metformin)", 0.85), ("DPP-4 inhibitors", 0.35), ("SGLT2 inhibitors", 0.20)],
    "I10": [("ACE inhibitors", 0.50), ("Calcium channel blockers", 0.38)],
    "E78.5": [("Statins", 0.80)],
    "E55.9": [("Vitamin D supplements", 0.85)],
    "J45.909": [("Inhaled corticosteroids", 0.60), ("Short-acting bronchodilators", 0.55)],
    "J06.9": [("Analgesics & antipyretics", 0.60), ("Antibiotics", 0.18), ("Antihistamines", 0.20)],
    "J02.9": [("Antibiotics", 0.45), ("Analgesics & antipyretics", 0.60)],
    "H66.90": [("Antibiotics", 0.70), ("Analgesics & antipyretics", 0.50)],
    "N39.0": [("Antibiotics", 0.90)],
    "A09": [("Oral rehydration salts", 0.75)],
    "M54.50": [("Analgesics & antipyretics", 0.70), ("NSAIDs", 0.45)],
    "M79.10": [("Analgesics & antipyretics", 0.70), ("NSAIDs", 0.45)],
    "R51.9": [("Analgesics & antipyretics", 0.75)],
    "K29.70": [("Proton pump inhibitors", 0.85)],
    "L30.9": [("Topical corticosteroids", 0.70), ("Antihistamines", 0.30)],
    "R50.9": [("Analgesics & antipyretics", 0.80)],
    "B34.9": [("Analgesics & antipyretics", 0.80)],
    "K02.9": [("Analgesics & antipyretics", 0.35), ("Antibiotics", 0.12)],
    "K05.10": [("Analgesics & antipyretics", 0.30), ("Antibiotics", 0.10)],
    "H10.9": [("Antibiotic eye drops", 0.60)],
}


# ---------------------------------------------------------------------------- helpers


def _categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Sample one category per row from a (n, k) matrix of probabilities."""
    probs = np.asarray(probs, dtype=float)
    cum = np.cumsum(probs / probs.sum(axis=1, keepdims=True), axis=1)
    cum[:, -1] = 1.0
    r = rng.random(len(probs))[:, None]
    return (r > cum).sum(axis=1)


def _choice(rng: np.random.Generator, n: int, values, p) -> np.ndarray:
    return np.asarray(values)[rng.choice(len(values), size=n, p=np.asarray(p) / np.sum(p))]


def _ids(prefix: str, n: int, width: int = 7) -> np.ndarray:
    return np.char.add(prefix, np.char.zfill(np.arange(1, n + 1).astype(str), width))


def _day_index(d: date) -> int:
    return (d - START).days


def _calendar() -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    cal = pd.DataFrame({"date": days})
    cal["year"] = days.year
    cal["quarter"] = days.quarter
    cal["month"] = days.month
    cal["month_name"] = days.month_name()
    cal["day_name"] = days.day_name()
    cal["is_weekend"] = days.dayofweek.isin([4, 5])  # Friday, Saturday
    in_range = lambda spans: np.any([(days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b)) for a, b in spans], axis=0)
    cal["is_ramadan"] = in_range(RAMADAN)
    cal["is_eid"] = in_range(EID)
    cal["is_summer"] = days.month.isin([6, 7, 8])
    return cal


def _day_weights(cal: pd.DataFrame) -> np.ndarray:
    """Relative booking volume per day: weekends, summer travel, Ramadan and Eid all lower it."""
    w = np.ones(len(cal))
    w[cal["day_name"].eq("Friday").to_numpy()] = 0.12
    w[cal["day_name"].eq("Saturday").to_numpy()] = 0.65
    w[cal["month"].isin([7, 8]).to_numpy()] *= 0.70
    w[cal["month"].eq(6).to_numpy()] *= 0.90
    w[cal["is_ramadan"].to_numpy()] *= 0.82
    w[cal["is_eid"].to_numpy()] *= 0.08
    w *= 1 + 0.06 * np.arange(len(cal)) / 365.0  # the network grows slowly
    return w


def _sample_days(rng, cum_w: np.ndarray, start_idx: np.ndarray, end_idx: np.ndarray | None = None) -> np.ndarray:
    """Sample a day index in [start_idx, end_idx] for each row, weighted by the day weights."""
    start_idx = np.asarray(start_idx)
    lo = np.where(start_idx > 0, cum_w[np.maximum(start_idx - 1, 0)], 0.0)
    hi = cum_w[-1] if end_idx is None else cum_w[np.asarray(end_idx)]
    u = lo + rng.random(len(start_idx)) * (hi - lo)
    return np.minimum(np.searchsorted(cum_w, u, side="right"), len(cum_w) - 1)


# ---------------------------------------------------------------------------- tables


def _clinics() -> pd.DataFrame:
    df = pd.DataFrame(
        [c[:9] for c in CLINICS],
        columns=["clinic_id", "clinic_name", "clinic_name_ar", "municipality", "opens_at", "closes_at",
                 "ramadan_opens_at", "ramadan_closes_at", "has_lab"],
    )
    for col in ["opens_at", "closes_at", "ramadan_opens_at", "ramadan_closes_at"]:
        df[col] = df[col] + ":00"
    return df


def _patients(rng: np.random.Generator, n: int, seed: int) -> tuple[pd.DataFrame, dict]:
    nat = rng.choice(len(NATIONALITIES), size=n, p=NAT_P)

    age_p = np.tile([0.18, 0.20, 0.25, 0.18, 0.12, 0.07], (n, 1))
    age_p[nat == QATARI] = [0.30, 0.20, 0.15, 0.13, 0.11, 0.11]
    age_p[nat == SOUTH_ASIAN] = [0.04, 0.30, 0.33, 0.20, 0.10, 0.03]
    age_p[nat == WESTERN] = [0.20, 0.15, 0.28, 0.20, 0.11, 0.06]
    age = _categorical(rng, age_p)

    p_male = np.full(n, 0.58)
    p_male[nat == SOUTH_ASIAN] = 0.80
    p_male[nat == SE_ASIAN] = 0.45
    p_male[nat == QATARI] = 0.49
    p_male[nat == AFRICAN] = 0.65
    p_male[age == KID] = 0.51
    male = rng.random(n) < p_male

    clinic_p = np.tile([0.20, 0.13, 0.20, 0.14, 0.08, 0.05, 0.10, 0.10], (n, 1))
    clinic_p[nat == SOUTH_ASIAN] = [0.10, 0.05, 0.12, 0.08, 0.04, 0.03, 0.50, 0.08]
    clinic_p[nat == QATARI] = [0.12, 0.10, 0.20, 0.15, 0.15, 0.10, 0.00, 0.18]
    clinic_p[nat == WESTERN] = [0.25, 0.45, 0.10, 0.08, 0.02, 0.02, 0.00, 0.08]
    clinic = _categorical(rng, clinic_p)
    municipality = CLINIC_MUNICIPALITY[clinic].copy()
    moved = rng.random(n) < 0.10
    municipality[moved] = rng.choice(np.unique(CLINIC_MUNICIPALITY), size=moved.sum())

    # 72% registered before the data period, the rest during it
    before = rng.random(n) < 0.72
    reg = np.empty(n, dtype="datetime64[D]")
    span_old = (np.datetime64("2022-12-31") - np.datetime64("2012-01-01")).astype(int)
    span_new = (np.datetime64("2025-09-30") - np.datetime64("2023-01-01")).astype(int)
    reg[before] = np.datetime64("2012-01-01") + rng.integers(0, span_old + 1, before.sum())
    reg[~before] = np.datetime64("2023-01-01") + rng.integers(0, span_new + 1, (~before).sum())

    # Hidden health profile (not stored as columns; it drives the events below)
    nat_mult = np.array([1.25, 1.10, 1.30, 0.90, 0.85, 0.55, 1.00])[nat]
    diabetic = rng.random(n) < np.clip(np.array([0.004, 0.03, 0.08, 0.17, 0.27, 0.36])[age] * nat_mult, 0, 0.6)
    htn = rng.random(n) < np.clip(
        np.array([0.003, 0.04, 0.10, 0.21, 0.33, 0.47])[age] * np.where(nat == WESTERN, 0.8, 1.0) + 0.15 * diabetic, 0, 0.8
    )
    lipid = rng.random(n) < np.clip(0.04 + 0.40 * diabetic + np.array([0.0, 0.03, 0.08, 0.15, 0.22, 0.28])[age], 0, 0.8)
    asthma = rng.random(n) < np.where(age == KID, 0.10, 0.07)
    vitd_p = np.full(n, 0.55)
    vitd_p[np.isin(nat, [QATARI, OTHER_ARAB])] = 0.68
    vitd_p[nat == WESTERN] = 0.35
    vitd_p += 0.08 * ~male
    vitd_def = rng.random(n) < vitd_p
    hba1c_base = np.clip(rng.normal(7.9, 1.3, n), 6.0, 12.5)
    engagement = rng.normal(0, 1, n)

    patient_id = np.array(
        ["P" + hashlib.sha256(f"atd-{seed}-{i}".encode()).hexdigest()[:10] for i in range(n)]
    )
    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "sex": np.where(male, "Male", "Female"),
            "age_band": AGE_BANDS[age],
            "nationality_group": NATIONALITIES[nat],
            "municipality": municipality,
            "registered_clinic_id": CLINIC_IDS[clinic],
            "registration_date": reg,
        }
    )
    hidden = dict(nat=nat, age=age, male=male, clinic=clinic, reg=reg, diabetic=diabetic, htn=htn, lipid=lipid,
                  asthma=asthma, vitd_def=vitd_def, hba1c_base=hba1c_base, engagement=engagement)
    return df, hidden


def generate(n_patients: int = 60_000, seed: int = 42, data_dir: Path | None = None, verbose: bool = True) -> dict:
    """Generate all tables, write Parquet files and build the DuckDB database."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir) if data_dir else Settings.from_env().data_dir
    parquet_dir = data_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    cal = _calendar()
    is_ramadan = cal["is_ramadan"].to_numpy()
    month = cal["month"].to_numpy()
    cum_w = np.cumsum(_day_weights(cal))

    clinics = _clinics()
    patients, h = _patients(rng, n_patients, seed)
    nat, age, male, home = h["nat"], h["age"], h["male"], h["clinic"]

    # ------------------------------------------------------------------ appointments
    start_idx = np.clip((h["reg"] - np.datetime64(START)).astype(int), 0, N_DAYS - 1)
    years_active = (N_DAYS - start_idx) / 365.25
    lam = (
        1.8
        + 0.6 * (age == KID)
        + 0.8 * (age == A60)
        + 3.2 * h["diabetic"]
        + 1.5 * h["htn"]
        + 0.4 * h["asthma"]
        + 0.3 * (~male & (age != KID))
        + 0.5 * (nat == QATARI)
        - 0.3 * (nat == SOUTH_ASIAN)
    )
    n_appt = rng.poisson(lam * years_active)
    pi = np.repeat(np.arange(n_patients), n_appt)  # patient index per appointment
    n_a = len(pi)
    day = _sample_days(rng, cum_w, start_idx[pi])

    a_age, a_nat, a_male, a_diab = age[pi], nat[pi], male[pi], h["diabetic"][pi]
    dept_p = np.tile([0.80, 0.0, 0.0, 0.0, 0.13, 0.07], (n_a, 1))
    female_adult = ~a_male & (a_age != KID)
    dept_p[female_adult & (a_age <= A40)] = [0.55, 0.0, 0.28, 0.0, 0.11, 0.06]
    dept_p[female_adult & (a_age >= A50)] = [0.70, 0.0, 0.12, 0.0, 0.11, 0.07]
    dept_p[a_diab] = [0.40, 0.45, 0.0, 0.0, 0.08, 0.07]
    dept_p[a_diab & female_adult] = [0.35, 0.42, 0.10, 0.0, 0.07, 0.06]
    dept_p[a_age == KID] = [0.05, 0.0, 0.0, 0.72, 0.12, 0.11]
    dept = _categorical(rng, dept_p)

    # Most visits happen at the home clinic; some patients use another clinic
    a_clinic = home[pi].copy()
    elsewhere = rng.random(n_a) < 0.12
    a_clinic[elsewhere] = rng.integers(0, len(CLINICS), elsewhere.sum())

    chan_p = np.tile([0.42, 0.36, 0.12, 0.10], (n_a, 1))
    chan_p[np.isin(a_nat, [QATARI, WESTERN])] = [0.60, 0.25, 0.08, 0.07]
    chan_p[a_nat == SOUTH_ASIAN] = [0.25, 0.40, 0.25, 0.10]
    chan_p[dept == VAC] = [0.35, 0.25, 0.35, 0.05]
    chan_p[dept == DC] = [0.40, 0.30, 0.05, 0.25]
    channel = _categorical(rng, chan_p)

    lead = np.zeros(n_a, dtype=int)
    for ch, mean in [(APP, 5), (PHONE, 8), (REFERRAL, 18)]:
        m = channel == ch
        lead[m] = rng.geometric(1 / (mean + 1), m.sum()) - 1
    lead = np.clip(lead, 0, 90)

    ram = is_ramadan[day]
    slot = np.where(
        ram,
        np.where(rng.random(n_a) < 0.5, rng.integers(36, 60, n_a), rng.integers(76, 92, n_a)),  # 09:00-14:45 / 19:00-22:45
        rng.integers(28, 84, n_a),  # 07:00-20:45
    )
    slot = np.where((a_clinic == INDUSTRIAL) & ~ram & (rng.random(n_a) < 0.15), rng.integers(24, 28, n_a), slot)
    minutes = slot * 15
    appt_time = np.char.add(
        np.char.add(np.char.zfill((minutes // 60).astype(str), 2), ":"),
        np.char.add(np.char.zfill((minutes % 60).astype(str), 2), ":00"),
    )

    summer = np.isin(month[day], [7, 8])
    eng = np.clip(h["engagement"][pi], -2, 2)
    p_noshow = (
        0.085
        + 0.065 * ram
        + 0.030 * summer
        + 0.050 * (lead > 14)
        + 0.035 * (a_age == A18)
        - 0.025 * (a_age == A60)
        + 0.050 * (a_clinic == INDUSTRIAL)
        + 0.020 * (a_nat == SOUTH_ASIAN)
        - 0.020 * (channel == APP)
        + 0.015 * (channel == REFERRAL)
        - 0.030 * eng
    )
    p_noshow = np.where(channel == WALKIN, 0.0, np.clip(p_noshow, 0.01, 0.6))
    p_cancel = np.where(channel == WALKIN, 0.005, 0.045 + 0.02 * (channel == APP) + 0.01 * (lead > 14))
    r = rng.random(n_a)
    status = np.where(r < p_cancel, "cancelled", np.where(r < p_cancel + p_noshow, "no_show", "attended"))

    appt_dates = np.datetime64(START) + day.astype("timedelta64[D]")
    appointments = pd.DataFrame(
        {
            "appointment_id": _ids("A", n_a),
            "patient_id": patients["patient_id"].to_numpy()[pi],
            "clinic_id": CLINIC_IDS[a_clinic],
            "department": DEPARTMENTS[dept],
            "appointment_date": appt_dates,
            "appointment_time": appt_time,
            "booking_channel": CHANNELS[channel],
            "lead_days": lead,
            "status": status,
        }
    )

    # ------------------------------------------------------------------ visits
    att = np.flatnonzero(status == "attended")
    n_v = len(att)
    v_pi, v_dept, v_day, v_clinic = pi[att], dept[att], day[att], a_clinic[att]
    v_chronic = (h["diabetic"] | h["htn"] | h["lipid"] | h["asthma"])[v_pi]

    vt_p = np.tile([0.62, 0.26, 0.12, 0.0], (n_v, 1))  # New complaint, Follow-up, Check-up, Vaccination
    vt_p[(v_dept == GP) & v_chronic] = [0.42, 0.45, 0.13, 0.0]
    vt_p[v_dept == PED] = [0.55, 0.10, 0.35, 0.0]
    vt_p[v_dept == DC] = [0.0, 0.85, 0.15, 0.0]
    vt_p[v_dept == VAC] = [0.0, 0.0, 0.0, 1.0]
    visit_type = np.array(["New complaint", "Follow-up", "Check-up", "Vaccination"])[_categorical(rng, vt_p)]

    evening_ramadan = is_ramadan[v_day] & (slot[att] >= 76)
    wait_mean = CLINIC_WAIT[v_clinic] * np.where(evening_ramadan, 1.25, 1.0) * np.where(v_dept == VAC, 0.6, 1.0)
    wait_mean = wait_mean + 6 * (v_dept == DEN)
    wait = np.clip(np.round(rng.gamma(2.2, wait_mean / 2.2)), 1, 240)
    wait_missing = rng.random(n_v) < 0.02

    visit_ids = _ids("V", n_v)
    visits = pd.DataFrame(
        {
            "visit_id": visit_ids,
            "appointment_id": appointments["appointment_id"].to_numpy()[att],
            "patient_id": appointments["patient_id"].to_numpy()[att],
            "clinic_id": CLINIC_IDS[v_clinic],
            "visit_date": appt_dates[att],
            "department": DEPARTMENTS[v_dept],
            "visit_type": visit_type,
            "doctor_specialty": SPECIALTY[v_dept],
            "wait_minutes": pd.array(np.where(wait_missing, np.nan, wait), dtype="Int64"),
        }
    )

    # ------------------------------------------------------------------ diagnoses
    v_age, v_male = age[v_pi], male[v_pi]
    primary = np.empty(n_v, dtype=object)

    def assign(mask, codes, p):
        idx = np.flatnonzero(mask)
        primary[idx] = _choice(rng, len(idx), codes, p)

    assign(v_dept == VAC, ["Z23"], [1])
    assign(v_dept == DEN, ["K02.9", "K05.10", "K08.89"], [0.5, 0.3, 0.2])
    assign(v_dept == DC, ["E11.9", "E11.65"], [0.72, 0.28])
    assign(v_dept == PED, ["J06.9", "Z00.129", "A09", "H66.90", "L30.9", "J02.9", "R50.9"],
           [0.32, 0.22, 0.10, 0.10, 0.08, 0.10, 0.08])
    ped_asthma = (v_dept == PED) & h["asthma"][v_pi] & (rng.random(n_v) < 0.35)
    primary[ped_asthma] = "J45.909"
    wh = v_dept == WH
    assign(wh, ["Z34.90", "Z01.419", "N39.0", "N94.6", "E55.9"], [0.30, 0.28, 0.16, 0.14, 0.12])
    primary[wh & (v_age >= A40) & (primary == "Z34.90")] = "Z01.419"  # pregnancy visits only for ages under 40
    primary[wh & ~h["vitd_def"][v_pi] & (primary == "E55.9")] = "N39.0"
    gp = v_dept == GP
    assign(gp, ["J06.9", "Z00.00", "M54.50", "J02.9", "K29.70", "R51.9", "L30.9", "N39.0", "M79.10", "R50.9", "B34.9", "H10.9"],
           [0.22, 0.14, 0.11, 0.09, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.04])
    conditions = np.column_stack([h["diabetic"], h["htn"], h["lipid"], h["vitd_def"], h["asthma"]])[v_pi]
    chronic_codes = np.array(["E11.9", "I10", "E78.5", "E55.9", "J45.909"])
    pick = np.argmax(conditions * rng.random((n_v, 5)), axis=1)
    gp_chronic = gp & conditions[:, :3].any(axis=1) & (rng.random(n_v) < 0.55)
    primary[gp_chronic] = chronic_codes[pick[gp_chronic]]
    gp_minor = gp & ~gp_chronic & conditions[:, 3:].any(axis=1) & (rng.random(n_v) < 0.20)
    primary[gp_minor] = chronic_codes[pick[gp_minor]]

    dx_visit = [np.arange(n_v)]
    dx_code = [primary]
    for mask, code in [
        ((v_dept == DC) & h["htn"][v_pi] & (rng.random(n_v) < 0.60), "I10"),
        ((v_dept == DC) & h["lipid"][v_pi] & (rng.random(n_v) < 0.45), "E78.5"),
        (gp & h["diabetic"][v_pi] & (primary != "E11.9") & (rng.random(n_v) < 0.25), "E11.9"),
        (gp & h["htn"][v_pi] & (primary != "I10") & (rng.random(n_v) < 0.20), "I10"),
    ]:
        idx = np.flatnonzero(mask)
        dx_visit.append(idx)
        dx_code.append(np.full(len(idx), code, dtype=object))
    dx_visit = np.concatenate(dx_visit)
    dx_code = np.concatenate(dx_code).astype(str)
    order = np.argsort(dx_visit, kind="stable")
    dx_visit, dx_code = dx_visit[order], dx_code[order]
    meta = pd.DataFrame.from_dict(ICD10, orient="index", columns=["icd10_description", "icd10_chapter", "is_chronic"])
    diagnoses = pd.DataFrame(
        {
            "diagnosis_id": _ids("D", len(dx_visit)),
            "visit_id": visit_ids[dx_visit],
            "patient_id": visits["patient_id"].to_numpy()[dx_visit],
            "visit_date": visits["visit_date"].to_numpy()[dx_visit],
            "icd10_code": dx_code,
        }
    ).join(meta, on="icd10_code")

    # ------------------------------------------------------------------ lab results
    v_years = v_day / 365.25
    v_diab = h["diabetic"][v_pi]
    adult40 = v_age >= A40
    hb_mask = (
        ((v_dept == DC) & (rng.random(n_v) < 0.80))
        | ((v_dept == GP) & v_diab & (rng.random(n_v) < 0.15))
        | ((visit_type == "Check-up") & adult40 & ~v_diab & (rng.random(n_v) < 0.25))
    )
    trend = np.where(h["engagement"] > 0, -0.18, 0.05)[v_pi]
    hb_val = np.where(
        v_diab,
        h["hba1c_base"][v_pi] + trend * v_years + rng.normal(0, 0.45, n_v),
        rng.normal(5.5, 0.35, n_v),
    )
    vd_p = np.where(v_age == KID, 0.03, 0.10)
    vd_p = np.where(v_dept == WH, 0.20, vd_p)
    vd_mask = ~np.isin(v_dept, [DEN, VAC]) & (rng.random(n_v) < vd_p)
    vd_val = np.where(
        h["vitd_def"][v_pi], np.clip(rng.normal(14, 4.5, n_v), 4, 40), np.clip(rng.normal(31, 7, n_v), 12, 80)
    )
    v_lipid = h["lipid"][v_pi]
    ldl_mask = (
        ((v_dept == DC) & (rng.random(n_v) < 0.35))
        | ((v_dept == GP) & v_lipid & (rng.random(n_v) < 0.25))
        | ((visit_type == "Check-up") & adult40 & (rng.random(n_v) < 0.15))
    )
    ldl_val = np.where(v_lipid, rng.normal(150, 28, n_v), rng.normal(108, 22, n_v))

    lab_parts = []
    for name, mask, values, unit, decimals in [
        ("HbA1c", hb_mask, np.clip(hb_val, 4.8, 14.0), "%", 1),
        ("Vitamin D", vd_mask, vd_val, "ng/mL", 1),
        ("LDL", ldl_mask, np.clip(ldl_val, 40, 300), "mg/dL", 0),
    ]:
        idx = np.flatnonzero(mask)
        lab_parts.append(
            pd.DataFrame(
                {"vi": idx, "test_name": name, "value": np.round(values[idx], decimals), "unit": unit}
            )
        )
    labs = pd.concat(lab_parts, ignore_index=True).sort_values(["vi", "test_name"], kind="stable")
    labs["patient_id"] = visits["patient_id"].to_numpy()[labs["vi"]]
    labs["visit_id"] = visit_ids[labs["vi"]]
    labs["test_date"] = visits["visit_date"].to_numpy()[labs["vi"]]
    labs = labs.drop_duplicates(["patient_id", "test_name", "test_date"]).reset_index(drop=True)
    labs.loc[rng.random(len(labs)) < 0.01, "unit"] = None
    lab_results = labs.assign(lab_id=_ids("L", len(labs)))[
        ["lab_id", "patient_id", "visit_id", "test_date", "test_name", "value", "unit"]
    ]

    # ------------------------------------------------------------------ prescriptions
    rx_parts = []
    dx_df = pd.DataFrame({"vi": dx_visit, "code": dx_code})
    for code, rules in PRESCRIBING.items():
        rows = dx_df.loc[dx_df["code"] == code, "vi"].to_numpy()
        for drug, p in rules:
            hit = rows[rng.random(len(rows)) < p]
            rx_parts.append(pd.DataFrame({"vi": hit, "drug_class": drug}))
        if code.startswith("E11"):  # insulin for poorly controlled diabetes
            base = h["hba1c_base"][v_pi[rows]]
            hit = rows[rng.random(len(rows)) < 0.08 + 0.25 * (base > 9)]
            rx_parts.append(pd.DataFrame({"vi": hit, "drug_class": "Insulin"}))
    low_vd = lab_results.loc[(lab_results["test_name"] == "Vitamin D") & (lab_results["value"] < 20), "visit_id"]
    low_vd_vi = np.searchsorted(visit_ids, low_vd.to_numpy())
    hit = low_vd_vi[rng.random(len(low_vd_vi)) < 0.70]
    rx_parts.append(pd.DataFrame({"vi": hit, "drug_class": "Vitamin D supplements"}))
    rx = pd.concat(rx_parts, ignore_index=True).drop_duplicates().sort_values(["vi", "drug_class"]).reset_index(drop=True)
    chronic_drug = rx["drug_class"].isin(CHRONIC_DRUGS).to_numpy()
    rx["days_supplied"] = np.where(
        chronic_drug,
        rng.choice([30, 90], size=len(rx), p=[0.6, 0.4]),
        rng.choice([3, 5, 7, 10], size=len(rx), p=[0.2, 0.35, 0.35, 0.1]),
    )
    prescriptions = pd.DataFrame(
        {
            "prescription_id": _ids("R", len(rx)),
            "visit_id": visit_ids[rx["vi"]],
            "patient_id": visits["patient_id"].to_numpy()[rx["vi"]],
            "prescribed_date": visits["visit_date"].to_numpy()[rx["vi"]],
            "drug_class": rx["drug_class"].to_numpy(),
            "days_supplied": rx["days_supplied"].to_numpy(),
        }
    )

    # ------------------------------------------------------------------ screenings
    sc_parts = []
    reg_idx = (h["reg"] - np.datetime64(START)).astype(int)
    for year in (2023, 2024, 2025):
        y0, y1 = _day_index(date(year, 1, 1)), _day_index(date(year, 12, 31))
        registered = reg_idx <= y1
        first_day = np.clip(reg_idx, y0, y1)
        p_diab = np.select([nat == QATARI, nat == WESTERN, nat == SOUTH_ASIAN], [0.42, 0.38, 0.28], 0.30)
        rules = [
            ("Diabetes", ((age >= A40) | ((nat == SOUTH_ASIAN) & (age == A30))) & ~h["diabetic"], p_diab,
             0.11 + 0.06 * (age >= A50)),
            ("Blood pressure", age >= A18, np.full(n_patients, 0.38), 0.12 + 0.55 * h["htn"]),
            ("Breast cancer", ~male & (age >= A40), np.where(age == A40, 0.12, 0.28), np.full(n_patients, 0.06)),
            ("Colorectal cancer", age >= A50, np.full(n_patients, 0.18), np.full(n_patients, 0.05)),
        ]
        for name, eligible, p_take, p_abn in rules:
            take = np.flatnonzero(registered & eligible & (rng.random(n_patients) < p_take))
            sday = _sample_days(rng, cum_w, first_day[take], np.full(len(take), y1))
            abnormal = rng.random(len(take)) < np.broadcast_to(p_abn, (n_patients,))[take]
            sc_parts.append(pd.DataFrame({"pi": take, "day": sday, "screening_type": name,
                                          "result": np.where(abnormal, "Abnormal", "Normal")}))
    sc = pd.concat(sc_parts, ignore_index=True).sort_values(["day", "pi"], kind="stable").reset_index(drop=True)
    screenings = pd.DataFrame(
        {
            "screening_id": _ids("S", len(sc)),
            "patient_id": patients["patient_id"].to_numpy()[sc["pi"]],
            "clinic_id": CLINIC_IDS[home[sc["pi"]]],
            "screening_date": np.datetime64(START) + sc["day"].to_numpy().astype("timedelta64[D]"),
            "screening_type": sc["screening_type"].to_numpy(),
            "result": sc["result"].to_numpy(),
        }
    )

    # ------------------------------------------------------------------ follow-ups
    has_i10 = np.zeros(n_v, dtype=bool)
    has_i10[dx_visit[dx_code == "I10"]] = True
    has_z34 = np.zeros(n_v, dtype=bool)
    has_z34[dx_visit[dx_code == "Z34.90"]] = True
    fu_parts = []
    for mask, reason, days_after in [
        ((v_dept == DC) & (rng.random(n_v) < 0.90), "Diabetes review", 90),
        (gp & has_i10 & (rng.random(n_v) < 0.60), "Blood pressure review", 90),
        (has_z34 & (rng.random(n_v) < 0.95), "Antenatal visit", 28),
    ]:
        idx = np.flatnonzero(mask)
        fu_parts.append(pd.DataFrame({"pi": v_pi[idx], "clinic": v_clinic[idx], "vi": idx,
                                      "due": v_day[idx] + days_after, "reason": reason, "source": "Visit"}))
    vd_idx = low_vd_vi[rng.random(len(low_vd_vi)) < 0.35]
    fu_parts.append(pd.DataFrame({"pi": v_pi[vd_idx], "clinic": v_clinic[vd_idx], "vi": vd_idx,
                                  "due": v_day[vd_idx] + 90, "reason": "Vitamin D recheck", "source": "Visit"}))
    abn = np.flatnonzero((sc["result"] == "Abnormal").to_numpy() & (rng.random(len(sc)) < 0.95))
    fu_parts.append(pd.DataFrame({"pi": sc["pi"].to_numpy()[abn], "clinic": home[sc["pi"].to_numpy()[abn]], "vi": -1,
                                  "due": sc["day"].to_numpy()[abn] + 30, "reason": "Abnormal screening result",
                                  "source": "Screening"}))
    fu = pd.concat(fu_parts, ignore_index=True)
    fu = fu[fu["due"] <= N_DAYS - 1].sort_values(["due", "pi", "reason"], kind="stable").reset_index(drop=True)
    f_pi, f_due = fu["pi"].to_numpy(), fu["due"].to_numpy()
    f_month = month[f_due]
    p_done = (
        0.70
        + 0.07 * (nat[f_pi] == QATARI)
        + 0.05 * (nat[f_pi] == WESTERN)
        - 0.14 * (nat[f_pi] == SOUTH_ASIAN)
        - 0.06 * (nat[f_pi] == SE_ASIAN)
        - 0.07 * (age[f_pi] == A18)
        + 0.05 * (age[f_pi] == A60)
        - 0.12 * np.isin(f_month, [7, 8])
        - 0.05 * is_ramadan[f_due]
        + 0.15 * (fu["reason"] == "Antenatal visit").to_numpy()
        + 0.06 * (fu["reason"] == "Abnormal screening result").to_numpy()
        + 0.06 * np.clip(h["engagement"][f_pi], -2, 2)
    )
    done = rng.random(len(fu)) < np.clip(p_done, 0.05, 0.97)
    done_day = f_due + np.clip(np.round(rng.normal(3, 9, len(fu))), -14, 45).astype(int)
    done &= done_day <= N_DAYS - 1
    due_dates = np.datetime64(START) + f_due.astype("timedelta64[D]")
    completed = np.datetime64(START) + done_day.astype("timedelta64[D]")
    vi = fu["vi"].to_numpy()
    follow_ups = pd.DataFrame(
        {
            "follow_up_id": _ids("F", len(fu)),
            "patient_id": patients["patient_id"].to_numpy()[f_pi],
            "clinic_id": CLINIC_IDS[fu["clinic"].to_numpy()],
            "source": fu["source"].to_numpy(),
            "source_visit_id": np.where(vi >= 0, visit_ids[np.maximum(vi, 0)], None),
            "reason": fu["reason"].to_numpy(),
            "due_date": due_dates,
            "completed_date": pd.Series(completed).where(done, pd.NaT).to_numpy(),
        }
    )

    tables = {
        "clinics": clinics,
        "patients": patients,
        "appointments": appointments,
        "visits": visits,
        "diagnoses": diagnoses,
        "lab_results": lab_results,
        "prescriptions": prescriptions,
        "screenings": screenings,
        "follow_ups": follow_ups,
        "calendar": cal,
    }
    _write(tables, data_dir)
    counts = {name: len(df) for name, df in tables.items()}
    if verbose:
        total = sum(counts.values())
        for name, c in counts.items():
            print(f"  {name:<14}{c:>12,}")
        print(f"  {'total':<14}{total:>12,}   ({time.perf_counter() - t0:.1f} s, seed {seed})")
    return counts


def _write(tables: dict[str, pd.DataFrame], data_dir: Path) -> None:
    parquet_dir = data_dir / "parquet"
    db_path = data_dir / "clinic.duckdb"
    for suffix in ("", ".wal"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for name, df in tables.items():
            spec = TABLES[name]
            missing = set(spec.column_names) - set(df.columns)
            if missing:
                raise RuntimeError(f"{name} is missing columns {missing}")
            path = parquet_dir / f"{name}.parquet"
            df[spec.column_names].to_parquet(path, index=False)
            casts = ", ".join(f'CAST("{c.name}" AS {c.type}) AS "{c.name}"' for c in spec.columns)
            con.execute(f"CREATE TABLE {name} AS SELECT {casts} FROM read_parquet(?)", [str(path)])
        con.execute("CHECKPOINT")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic clinic data mart.")
    parser.add_argument("--patients", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir or Settings.from_env().data_dir
    print(f"Generating {args.patients:,} patients into {data_dir} ...")
    generate(args.patients, args.seed, data_dir)


if __name__ == "__main__":
    main()
