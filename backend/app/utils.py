from __future__ import annotations

from typing import Iterable


def build_recommendations(employee: dict, predicted_band: str) -> list[str]:
    actions: list[str] = []

    if employee.get("on_time_delivery_rate", 1) < 0.75:
        actions.append("Improve sprint planning and time management with weekly milestone reviews.")
    if employee.get("training_hours", 0) < 20:
        actions.append("Assign targeted upskilling modules and track training completion monthly.")
    if employee.get("peer_feedback_score", 5) < 3.5:
        actions.append("Schedule peer collaboration coaching and communication feedback sessions.")
    if employee.get("manager_score", 5) < 3.5:
        actions.append("Conduct one-on-one manager coaching with a 30-day improvement plan.")
    if employee.get("bug_count", 0) > 20 or employee.get("qa_defect_density", 0) > 3:
        actions.append("Provide code quality mentoring, root-cause review, and QA best-practice support.")
    if employee.get("unplanned_absences", 0) > 3:
        actions.append("Review attendance blockers and create a support-based attendance improvement plan.")
    if employee.get("engagement_score", 100) < 60:
        actions.append("Run engagement check-ins and assign work aligned to strengths and growth goals.")
    if employee.get("kudos_count", 0) < 3 and predicted_band != "Low":
        actions.append("Increase recognition through peer appreciation and visible milestone tracking.")

    if predicted_band == "High":
        actions.append("Consider leadership pipeline, promotion readiness review, or advanced stretch assignments.")
    elif predicted_band == "Medium":
        actions.append("Set a focused 60-day development plan to move performance into the high band.")
    else:
        actions.append("Start an early support intervention with measurable goals, coaching, and training checkpoints.")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in actions:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped[:5]


def round_probabilities(classes: Iterable[str], probs: Iterable[float]) -> dict[str, float]:
    return {str(label): round(float(prob), 4) for label, prob in zip(classes, probs)}
