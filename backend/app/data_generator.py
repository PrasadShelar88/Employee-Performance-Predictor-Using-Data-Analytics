from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATASET_PATH, DATA_DIR


DEPARTMENTS = ["Engineering", "HR", "Sales", "Finance", "Marketing", "Support"]
EDUCATION_LEVELS = ["Diploma", "Bachelor", "Master", "PhD"]
JOB_LEVELS = ["Junior", "Mid", "Senior", "Lead"]
GENDERS = ["Male", "Female", "Other"]
SALARY_BANDS = ["Low", "Medium", "High"]


def _clip_round(values, low, high, decimals=2):
    values = np.clip(values, low, high)
    return np.round(values, decimals)


def generate_synthetic_employee_data(n_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 60, size=n_rows)
    gender = rng.choice(GENDERS, size=n_rows, p=[0.5, 0.47, 0.03])
    education_level = rng.choice(EDUCATION_LEVELS, size=n_rows, p=[0.12, 0.52, 0.28, 0.08])
    department = rng.choice(DEPARTMENTS, size=n_rows, p=[0.3, 0.1, 0.18, 0.12, 0.12, 0.18])
    job_level = rng.choice(JOB_LEVELS, size=n_rows, p=[0.3, 0.35, 0.25, 0.1])

    experience_years = np.clip(age - rng.integers(21, 27, size=n_rows), 0, 35)
    manager_tenure = _clip_round(rng.normal(4.5, 2.2, size=n_rows), 0, 20, 1)
    training_hours = _clip_round(rng.normal(38, 18, size=n_rows), 0, 120, 1)
    certifications_count = np.clip(rng.poisson(2.2, size=n_rows), 0, 12)
    projects_count = np.clip(rng.poisson(5, size=n_rows) + 1, 1, 15)
    avg_task_delay_days = _clip_round(rng.normal(4, 4, size=n_rows), -5, 25, 1)
    on_time_delivery_rate = _clip_round(rng.beta(8, 2, size=n_rows), 0.35, 1, 3)
    bug_count = np.clip(rng.poisson(12, size=n_rows), 0, 80)
    code_review_score = _clip_round(rng.normal(3.7, 0.7, size=n_rows), 1, 5, 2)
    qa_defect_density = _clip_round(rng.normal(1.8, 0.9, size=n_rows), 0, 8, 2)
    story_points_completed = _clip_round(rng.normal(65, 22, size=n_rows), 10, 180, 1)
    billable_hours_ratio = _clip_round(rng.beta(7, 2.5, size=n_rows), 0.2, 1, 3)
    sick_days = np.clip(rng.poisson(4, size=n_rows), 0, 20)
    unplanned_absences = np.clip(rng.poisson(2, size=n_rows), 0, 12)
    avg_login_hours = _clip_round(rng.normal(8.1, 1.2, size=n_rows), 4, 12, 1)
    peer_feedback_score = _clip_round(rng.normal(3.8, 0.6, size=n_rows), 1.5, 5, 2)
    manager_score = _clip_round(rng.normal(3.7, 0.7, size=n_rows), 1, 5, 2)
    kudos_count = np.clip(rng.poisson(8, size=n_rows), 0, 80)
    promotions_in_2y = np.clip(rng.binomial(2, 0.18, size=n_rows), 0, 2)
    salary_percentile_band = rng.choice(SALARY_BANDS, size=n_rows, p=[0.35, 0.45, 0.2])
    internal_hackathons_participated = np.clip(rng.poisson(1.2, size=n_rows), 0, 8)
    overtime_hours_monthly = _clip_round(rng.normal(16, 10, size=n_rows), 0, 60, 1)
    engagement_score = _clip_round(rng.normal(70, 15, size=n_rows), 20, 100, 1)
    last_promotion_years_ago = _clip_round(rng.normal(3.0, 1.8, size=n_rows), 0, 12, 1)

    job_level_bonus = pd.Series(job_level).map({"Junior": 0, "Mid": 2.5, "Senior": 4.5, "Lead": 6.0}).to_numpy()
    education_bonus = pd.Series(education_level).map({"Diploma": 0, "Bachelor": 1.2, "Master": 2.3, "PhD": 3.0}).to_numpy()
    salary_base = (
        3.0 + experience_years * 0.45 + job_level_bonus + education_bonus + pd.Series(salary_percentile_band).map({"Low": 0, "Medium": 2.5, "High": 5.5}).to_numpy()
    )
    salary_lpa = _clip_round(salary_base + rng.normal(0, 1.5, size=n_rows), 2.5, 35.0, 2)

    score = (
        0.06 * experience_years
        + 1.4 * on_time_delivery_rate
        + 0.28 * code_review_score
        + 0.32 * peer_feedback_score
        + 0.35 * manager_score
        + 0.02 * training_hours
        + 0.06 * certifications_count
        + 0.02 * story_points_completed
        + 0.012 * engagement_score
        + 0.04 * kudos_count
        + 0.08 * promotions_in_2y
        + 0.05 * internal_hackathons_participated
        - 0.03 * bug_count
        - 0.14 * qa_defect_density
        - 0.06 * avg_task_delay_days
        - 0.04 * unplanned_absences
        - 0.02 * sick_days
        - 0.015 * overtime_hours_monthly
        + rng.normal(0, 0.35, size=n_rows)
    )

    q1, q2 = np.quantile(score, [0.33, 0.66])
    perf_band_next = np.where(score >= q2, "High", np.where(score >= q1, "Medium", "Low"))

    df = pd.DataFrame(
        {
            "employee_id": [f"EMP{10000+i}" for i in range(n_rows)],
            "age": age,
            "gender": gender,
            "education_level": education_level,
            "department": department,
            "job_level": job_level,
            "experience_years": experience_years,
            "salary_lpa": salary_lpa,
            "training_hours": training_hours,
            "certifications_count": certifications_count,
            "projects_count": projects_count,
            "avg_task_delay_days": avg_task_delay_days,
            "on_time_delivery_rate": on_time_delivery_rate,
            "bug_count": bug_count,
            "code_review_score": code_review_score,
            "qa_defect_density": qa_defect_density,
            "story_points_completed": story_points_completed,
            "billable_hours_ratio": billable_hours_ratio,
            "sick_days": sick_days,
            "unplanned_absences": unplanned_absences,
            "avg_login_hours": avg_login_hours,
            "peer_feedback_score": peer_feedback_score,
            "manager_score": manager_score,
            "kudos_count": kudos_count,
            "promotions_in_2y": promotions_in_2y,
            "salary_percentile_band": salary_percentile_band,
            "manager_tenure": manager_tenure,
            "internal_hackathons_participated": internal_hackathons_participated,
            "overtime_hours_monthly": overtime_hours_monthly,
            "engagement_score": engagement_score,
            "last_promotion_years_ago": last_promotion_years_ago,
            "perf_band_next": perf_band_next,
        }
    )
    return df


def save_dataset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = generate_synthetic_employee_data()
    df.to_csv(DATASET_PATH, index=False)
    return df


if __name__ == "__main__":
    dataset = save_dataset()
    print(f"Saved synthetic dataset to: {DATASET_PATH}")
    print(dataset.head())
