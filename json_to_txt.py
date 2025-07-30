def json_to_txt(results):
    if not results:
        return "No star detected for this threshold."

    lines = []
    for r in results:
        if r["class"] == "maybeStar":
            lines.append(
                f"Image contains noise with confidence {int(r['confidence']*100)}% "
                f"at position ({int(r['x'])}, {int(r['y'])}) "
                f"with width {int(r['width'])} and height {int(r['height'])}"
            )
        else:  # star
            lines.append(
                f"Image contains star with confidence {int(r['confidence']*100)}% "
                f"at position ({int(r['x'])}, {int(r['y'])}) "
                f"with width {int(r['width'])} and height {int(r['height'])} "
                f"and brightness {int(r['brightness'])}"
            )
    return "\n".join(lines)
