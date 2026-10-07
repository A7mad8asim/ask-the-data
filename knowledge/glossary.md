# Business glossary (English / العربية)

## Time
- The data covers 2023-01-01 to 2025-12-31. Treat "today" / "now" / "الحين" / "الآن" as 2025-12-31.
- "This year" / "هذه السنة" / "هالسنة" = 2025. "Last year" / "السنة الماضية" / "السنة اللي فاتت" = 2024.
- "Last quarter" / "الربع الأخير" / "الربع الماضي" = 2025 Q4 (2025-10-01 to 2025-12-31).
- "Last month" / "الشهر الماضي" = December 2025. "Last 6 months" = 2025-07-01 to 2025-12-31.
- Ramadan (رمضان), Eid (العيد), weekend (عطلة نهاية الأسبوع / الويكند: Friday and Saturday) and summer (الصيف: June to August): join `calendar` on the date and use `is_ramadan`, `is_eid`, `is_weekend`, `is_summer`. "Ramadan 2025" = `calendar.is_ramadan AND calendar.year = 2025`.
- Use `year(date_column)` and `month(date_column)` for years and months (month as a number 1 to 12).

## Appointments and visits
- Appointment (موعد / حجز) = a booked slot in `appointments`. Visit (زيارة) = an attended appointment, in `visits`.
- No-show (عدم الحضور / الغياب / "ما حضروا" / "ما جوا") = `status = 'no_show'`.
- No-show rate (نسبة عدم الحضور / نسبة الغياب) = no-shows ÷ (attended + no-shows) × 100. Cancelled appointments are excluded from the denominator.
- Attendance rate (نسبة الحضور) = attended ÷ (attended + no-shows) × 100.
- Cancellation rate (نسبة الإلغاء) = cancelled ÷ all appointments × 100. "Cancelled" / "تكنسل" / "انلغى" = `status = 'cancelled'`.
- Lead time (مدة الانتظار للموعد) = `appointments.lead_days`.
- Waiting time (وقت الانتظار) = `visits.wait_minutes`. It is NULL when not recorded; AVG and MEDIAN ignore NULLs.
- Booking channels (طريقة الحجز): 'App' (التطبيق), 'Phone' (الهاتف / التلفون), 'Walk-in' (حضور مباشر), 'Referral' (تحويل).
- Departments (الأقسام): 'General Practice' (الطب العام), 'Diabetes Clinic' (عيادة السكري), 'Women''s Health' (صحة المرأة), 'Pediatrics' (طب الأطفال), 'Dental' (الأسنان), 'Vaccination' (التطعيمات).
- Clinics: use `clinics.clinic_name` in results. Arabic names are in `clinics.clinic_name_ar` (for example عيادة المنطقة الصناعية = 'Industrial Area Clinic', عيادة الوكرة = 'Al Wakrah Clinic').

## Patients
- Count patients with `COUNT(DISTINCT patient_id)`; count events with `COUNT(*)`.
- Nationality / nationality group (الجنسية / مجموعة الجنسية / "جنسية") = `patients.nationality_group`. Sex (الجنس: رجال / ذكور = 'Male', نساء / حريم / إناث = 'Female') = `patients.sex`.
- Age band (الفئة العمرية) = `patients.age_band`: '0-17', '18-29', '30-39', '40-49', '50-59', '60+'. "Children" (الأطفال) = '0-17'. "Elderly" / "كبار السن" = '60+'.
- Registered clinic (عيادة التسجيل) = `patients.registered_clinic_id`. New patients registered in a year = `year(registration_date)`.

## Conditions and lab tests
- Diabetic patient (مريض السكري / مريض السكر) = a patient with at least one diagnosis where `icd10_code LIKE 'E11%'`.
- Hypertension (ارتفاع ضغط الدم / مريض الضغط) = `icd10_code = 'I10'`. High cholesterol (الكوليسترول) = 'E78.5'. Asthma (الربو) = `icd10_code LIKE 'J45%'`. Vitamin D deficiency diagnosis = 'E55.9'.
- HbA1c (السكر التراكمي) is in `lab_results` with `test_name = 'HbA1c'` (value in %).
- Controlled diabetes (السكري المنضبط) = the patient's latest HbA1c in the period is below 7.0. Poorly controlled (غير منضبط) = latest HbA1c above 9.0. For "latest", rank each patient's tests by `test_date DESC, lab_id DESC`.
- Vitamin D deficiency (نقص فيتامين د) = a 'Vitamin D' result below 20 ng/mL. Insufficiency = 20 to 29.9.
- High LDL (ارتفاع الكوليسترول الضار) = an 'LDL' result of 130 mg/dL or more.

## Screening and follow-up
- Screening types (الفحوصات): 'Diabetes' (فحص السكري), 'Blood pressure' (فحص ضغط الدم), 'Breast cancer' (فحص سرطان الثدي), 'Colorectal cancer' (فحص سرطان القولون). Abnormal (غير طبيعي) = `result = 'Abnormal'`.
- Follow-up completion rate (نسبة إكمال المتابعة) = follow-ups with `completed_date IS NOT NULL` ÷ follow-ups due in the period (filter on `due_date`) × 100.
- Completed on time (في الوقت المحدد) = `completed_date <= due_date + INTERVAL 14 DAY`.

## Drugs (drug_class values)
'Biguanides (metformin)', 'DPP-4 inhibitors', 'SGLT2 inhibitors', 'Insulin', 'ACE inhibitors', 'Calcium channel blockers', 'Statins' (الستاتين), 'Vitamin D supplements' (مكملات فيتامين د), 'Inhaled corticosteroids', 'Short-acting bronchodilators', 'Antibiotics' (المضادات الحيوية), 'Antibiotic eye drops', 'Analgesics & antipyretics', 'NSAIDs', 'Antihistamines', 'Proton pump inhibitors', 'Topical corticosteroids', 'Oral rehydration salts'.

## Gulf dialect words (لهجة خليجية)
- "شنو" / "ايش" / "شو" = what. "كم واحد" = how many people. "وايد" = a lot. "أكثر شي" = the most. "سوّى" = did / had (e.g. سوّى فحص = had a test). "انصرف له" = was prescribed. "طلع" = turned out / the result was. "الحين" = now.
