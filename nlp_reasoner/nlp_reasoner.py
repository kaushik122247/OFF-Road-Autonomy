
def generate_narration(stats, mission="safety"):
    """
    Improved reasoning logic with lower thresholds and more descriptive feedback.
    """
    rocks = stats.get("rocks", 0)
    logs = stats.get("logs", 0)
    sand = stats.get("sand", 0) # Ground clutter
    landscape = stats.get("landscape", 0)
    obstacles = stats.get("obstacles", 0)
    veg = stats.get("vegetation", 0)

    narration = []

    # 1. GENERAL TERRAIN DESCRIPTION (Always add a context line)
    if landscape > 30:
        narration.append("The path ahead looks largely clear and drivable.")
    elif landscape > 10:
        narration.append("The terrain is mixed with some drivable patches.")
    else:
        narration.append("Navigatable terrain is limited; proceed with caution.")

    # 2. MISSION SPECIFIC REASONING
    if mission == "speed":
        if obstacles > 5:
            narration.append(f"Obstacle density ({obstacles}%) will limit maximum velocity.")
        if rocks > 8:
            narration.append("Surface roughness from rocks may cause vibration at high speeds.")
        if landscape > 25:
            narration.append("Good visibility of the landscape allows for moderate speed increase.")
        else:
            narration.append("Limited clear path requires low-speed crawling.")

    elif mission == "energy":
        if sand > 15:
            narration.append("Soft ground clutter will increase rolling resistance and battery drain.")
        if rocks > 5:
            narration.append("Navigating around rocks requires frequent steering corrections, consuming more power.")
        if landscape > 40:
            narration.append("High percentage of flat landscape ($landscape}%) supports energy-efficient cruising.")

    elif mission == "safety":
        if obstacles > 8:
            narration.append(f"Caution: High concentration of hazardous obstacles ({obstacles}%) detected.")
        if rocks > 10:
            narration.append("The surface is highly uneven; risk of chassis damage is elevated.")
        if logs > 3:
            narration.append("Large logs detected which may pose high centering risks.")
        if veg > 20:
            narration.append("Dense vegetation may obscure smaller hazards.")

    elif mission == "exploration":
        if landscape > 30:
            narration.append("Good opportunities for mapping open areas.")
        if obstacles > 12:
            narration.append("Complex terrain structure provides high scientific interest for sampling.")
        if veg > 15:
            narration.append("Biological features (vegetation) are prominent in this sector.")

    # 3. FINAL FALLBACK (Just in case)
    if not narration:
        narration.append("Terrain conditions are stable for the current objective.")

    return " ".join(narration)

if __name__ == "__main__":
    dummy_stats = {"rocks": 8, "logs": 2, "sand": 15, "landscape": 20, "obstacles": 10, "vegetation": 5}
    print(generate_narration(dummy_stats, "safety"))
