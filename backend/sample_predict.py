import json
import pandas as pd

from app.ml import load_model
from app.utils import build_recommendations, round_probabilities

sample = {
    "age": 29,
    "gender": "Male",
    "education_level": "Bachelor",
    "department": "Engineering",
    "job_level": "Mid",
    "experience_years": 6,
    "salary_lpa": 9.8,
    "training_hours": 42,
    "certifications_count": 3,
    "projects_count": 5,
    "avg_task_delay_days": 2.0,
    "on_time_delivery_rate": 0.91,
    "bug_count": 7,
    "code_review_score": 4.2,
    "qa_defect_density": 1.1,
    "story_points_completed": 78.0,
    "billable_hours_ratio": 0.86,
    "sick_days": 2,
    "unplanned_absences": 1,
    "avg_login_hours": 8.5,
    "peer_feedback_score": 4.4,
    "manager_score": 4.3,
    "kudos_count": 12,
    "promotions_in_2y": 1,
    "salary_percentile_band": "Medium",
    "manager_tenure": 4.0,
    "internal_hackathons_participated": 2,
    "overtime_hours_monthly": 10.0,
    "engagement_score": 82.0,
    "last_promotion_years_ago": 1.5,
}

model = load_model()
df = pd.DataFrame([sample])
pred = model.predict(df)[0]
probs = round_probabilities(model.classes_, model.predict_proba(df)[0])
recommendations = build_recommendations(sample, pred)

print(json.dumps({
    "predicted_band": pred,
    "class_probabilities": probs,
    "recommended_actions": recommendations,
}, indent=2))
