"""Single source of truth for the data mart: tables, columns and what they mean.

The generator, the DuckDB build step, the LLM prompt and the tests all read from here,
so the description the model sees can never drift from the real tables.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    columns: tuple[Column, ...]
    patient_level: bool  # True when a row describes one patient or one of their events

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


def _cols(*rows: tuple[str, str, str]) -> tuple[Column, ...]:
    return tuple(Column(*r) for r in rows)


TABLES: dict[str, Table] = {
    t.name: t
    for t in [
        Table(
            "clinics",
            "One row per clinic in the fictional Doha primary-care network (8 clinics).",
            _cols(
                ("clinic_id", "VARCHAR", "Clinic code, 'C01' to 'C08'"),
                ("clinic_name", "VARCHAR", "English name, e.g. 'Doha Central Clinic', 'Industrial Area Clinic'"),
                ("clinic_name_ar", "VARCHAR", "Arabic name, e.g. 'عيادة الدوحة المركزية'"),
                ("municipality", "VARCHAR", "Municipality: 'Doha', 'Al Rayyan', 'Al Wakrah', 'Umm Salal', 'Al Khor', 'Al Daayen'"),
                ("opens_at", "TIME", "Normal opening time"),
                ("closes_at", "TIME", "Normal closing time"),
                ("ramadan_opens_at", "TIME", "Opening time during Ramadan"),
                ("ramadan_closes_at", "TIME", "Closing time during Ramadan"),
                ("has_lab", "BOOLEAN", "Whether the clinic has an on-site laboratory"),
            ),
            patient_level=False,
        ),
        Table(
            "patients",
            "One row per registered patient. Contains no names or contact details.",
            _cols(
                ("patient_id", "VARCHAR", "Hashed patient identifier (never show it in answers)"),
                ("sex", "VARCHAR", "'Male' or 'Female'"),
                ("age_band", "VARCHAR", "Age band as of 2025: '0-17', '18-29', '30-39', '40-49', '50-59', '60+'"),
                ("nationality_group", "VARCHAR", "'Qatari', 'Other Arab', 'South Asian', 'Southeast Asian', 'African', 'Western', 'Other'"),
                ("municipality", "VARCHAR", "Home municipality"),
                ("registered_clinic_id", "VARCHAR", "The patient's home clinic (joins to clinics.clinic_id)"),
                ("registration_date", "DATE", "Date the patient registered with the network"),
            ),
            patient_level=True,
        ),
        Table(
            "appointments",
            "One row per booked appointment slot, whether attended or not.",
            _cols(
                ("appointment_id", "VARCHAR", "Appointment identifier"),
                ("patient_id", "VARCHAR", "Joins to patients.patient_id"),
                ("clinic_id", "VARCHAR", "Joins to clinics.clinic_id"),
                ("department", "VARCHAR", "'General Practice', 'Diabetes Clinic', 'Women''s Health', 'Pediatrics', 'Dental', 'Vaccination'"),
                ("appointment_date", "DATE", "Date of the appointment"),
                ("appointment_time", "TIME", "Start time of the slot"),
                ("booking_channel", "VARCHAR", "'App', 'Phone', 'Walk-in', 'Referral'"),
                ("lead_days", "INTEGER", "Days between booking and the appointment date (0 for walk-ins)"),
                ("status", "VARCHAR", "'attended', 'no_show' or 'cancelled'"),
            ),
            patient_level=True,
        ),
        Table(
            "visits",
            "One row per attended appointment (a visit). Every visit links to one appointment.",
            _cols(
                ("visit_id", "VARCHAR", "Visit identifier"),
                ("appointment_id", "VARCHAR", "Joins to appointments.appointment_id"),
                ("patient_id", "VARCHAR", "Joins to patients.patient_id"),
                ("clinic_id", "VARCHAR", "Joins to clinics.clinic_id"),
                ("visit_date", "DATE", "Date of the visit (same as the appointment date)"),
                ("department", "VARCHAR", "Same values as appointments.department"),
                ("visit_type", "VARCHAR", "'New complaint', 'Follow-up', 'Check-up', 'Vaccination'"),
                ("doctor_specialty", "VARCHAR", "'Family Medicine', 'Endocrinology', 'Obstetrics & Gynaecology', 'Paediatrics', 'Dentistry', 'Nursing'"),
                ("wait_minutes", "INTEGER", "Minutes from check-in to being seen; NULL when not recorded"),
            ),
            patient_level=True,
        ),
        Table(
            "diagnoses",
            "One row per diagnosis recorded at a visit (a visit can have more than one).",
            _cols(
                ("diagnosis_id", "VARCHAR", "Diagnosis identifier"),
                ("visit_id", "VARCHAR", "Joins to visits.visit_id"),
                ("patient_id", "VARCHAR", "Joins to patients.patient_id"),
                ("visit_date", "DATE", "Date of the visit"),
                ("icd10_code", "VARCHAR", "ICD-10 code, e.g. 'E11.9' (type 2 diabetes), 'I10' (hypertension), 'E55.9' (vitamin D deficiency)"),
                ("icd10_description", "VARCHAR", "Description of the code"),
                ("icd10_chapter", "VARCHAR", "Chapter, e.g. 'Endocrine', 'Respiratory', 'Circulatory', 'Digestive'"),
                ("is_chronic", "BOOLEAN", "TRUE for long-term conditions (diabetes, hypertension, high cholesterol, asthma, vitamin D deficiency)"),
            ),
            patient_level=True,
        ),
        Table(
            "lab_results",
            "One row per lab test result.",
            _cols(
                ("lab_id", "VARCHAR", "Lab result identifier"),
                ("patient_id", "VARCHAR", "Joins to patients.patient_id"),
                ("visit_id", "VARCHAR", "Joins to visits.visit_id"),
                ("test_date", "DATE", "Date of the test"),
                ("test_name", "VARCHAR", "'HbA1c', 'Vitamin D' or 'LDL'"),
                ("value", "DOUBLE", "Result value (HbA1c in %, Vitamin D in ng/mL, LDL in mg/dL)"),
                ("unit", "VARCHAR", "'%', 'ng/mL' or 'mg/dL'; NULL in about 1% of rows (data-entry gap)"),
            ),
            patient_level=True,
        ),
        Table(
            "prescriptions",
            "One row per medicine class prescribed at a visit.",
            _cols(
                ("prescription_id", "VARCHAR", "Prescription identifier"),
                ("visit_id", "VARCHAR", "Joins to visits.visit_id"),
                ("patient_id", "VARCHAR", "Joins to patients.patient_id"),
                ("prescribed_date", "DATE", "Date prescribed"),
                ("drug_class", "VARCHAR", "e.g. 'Biguanides (metformin)', 'Insulin', 'Statins', 'Antibiotics', 'Vitamin D supplements'"),
                ("days_supplied", "INTEGER", "Days of medicine supplied"),
            ),
            patient_level=True,
        ),
        Table(
            "screenings",
            "One row per screening test done under the screening programme (not tied to a visit).",
            _cols(
                ("screening_id", "VARCHAR", "Screening identifier"),
                ("patient_id", "VARCHAR", "Joins to patients.patient_id"),
                ("clinic_id", "VARCHAR", "Clinic where the screening was done"),
                ("screening_date", "DATE", "Date of the screening"),
                ("screening_type", "VARCHAR", "'Diabetes', 'Blood pressure', 'Breast cancer', 'Colorectal cancer'"),
                ("result", "VARCHAR", "'Normal' or 'Abnormal'"),
            ),
            patient_level=True,
        ),
        Table(
            "follow_ups",
            "One row per planned follow-up (after a visit or an abnormal screening).",
            _cols(
                ("follow_up_id", "VARCHAR", "Follow-up identifier"),
                ("patient_id", "VARCHAR", "Joins to patients.patient_id"),
                ("clinic_id", "VARCHAR", "Clinic responsible for the follow-up"),
                ("source", "VARCHAR", "'Visit' or 'Screening'"),
                ("source_visit_id", "VARCHAR", "The visit that created it; NULL when the source is a screening"),
                ("reason", "VARCHAR", "'Diabetes review', 'Blood pressure review', 'Antenatal visit', 'Vitamin D recheck', 'Abnormal screening result'"),
                ("due_date", "DATE", "Date the follow-up is due"),
                ("completed_date", "DATE", "Date it was completed; NULL if not completed"),
            ),
            patient_level=True,
        ),
        Table(
            "calendar",
            "One row per day from 2023-01-01 to 2025-12-31. Join on the date to filter by Ramadan, Eid, weekend or summer.",
            _cols(
                ("date", "DATE", "The day"),
                ("year", "INTEGER", "Calendar year"),
                ("quarter", "INTEGER", "1 to 4"),
                ("month", "INTEGER", "1 to 12"),
                ("month_name", "VARCHAR", "'January' to 'December'"),
                ("day_name", "VARCHAR", "'Sunday' to 'Saturday'"),
                ("is_weekend", "BOOLEAN", "TRUE on Friday and Saturday (the weekend in Qatar)"),
                ("is_ramadan", "BOOLEAN", "TRUE during Ramadan"),
                ("is_eid", "BOOLEAN", "TRUE on Eid al-Fitr and Eid al-Adha public holidays"),
                ("is_summer", "BOOLEAN", "TRUE from June to August"),
            ),
            patient_level=False,
        ),
    ]
}

PATIENT_TABLES: frozenset[str] = frozenset(n for n, t in TABLES.items() if t.patient_level)

# Columns that point at one person or one event. They may only appear inside COUNT(...).
IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {
        "patient_id",
        "appointment_id",
        "visit_id",
        "diagnosis_id",
        "lab_id",
        "prescription_id",
        "screening_id",
        "follow_up_id",
        "source_visit_id",
    }
)


def schema_prompt() -> str:
    """Render the schema as compact text for the LLM prompt."""
    parts = []
    for table in TABLES.values():
        lines = [f"TABLE {table.name} -- {table.description}"]
        for c in table.columns:
            lines.append(f"  {c.name} {c.type} -- {c.description}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
