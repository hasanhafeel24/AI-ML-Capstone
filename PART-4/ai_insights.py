def generate_insights(score):
    """
    Generate simple AI-style feedback based on predicted score.
    """

    if score >= 85:
        return [
            "🌟 Excellent academic performance predicted.",
            "📚 Keep maintaining your current study routine.",
            "😴 Continue getting enough sleep.",
            "🎯 Stay consistent to achieve even better results."
        ]

    elif score >= 70:
        return [
            "👍 Good academic performance predicted.",
            "📖 Increase study time slightly.",
            "📝 Practice more mock tests.",
            "💪 Stay consistent with attendance."
        ]

    else:
        return [
            "⚠️ Improvement is needed.",
            "📚 Increase study hours.",
            "👨‍🏫 Attend tutoring sessions regularly.",
            "🎯 Focus on consistent daily practice."
        ]