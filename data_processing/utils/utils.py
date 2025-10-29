
def format_seconds(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    return "%d:%02d:%02d" % (hour, minutes, seconds)

explicit_exclude_list = ['no-half-measures-creating-the-final-season-of-breaking-bad', 'twin-peaks', 'fullmetal-alchemist-brotherhood', 'monster-2004', 'cowboy-bebop', 'one-piece-fan-letter', 'avatar-spirits', 'twin-peaks-the-return', 'attack-on-titan-the-last-attack', 'attack-on-titan-the-final-chapters-special-2', 'attack-on-titan-the-final-chapters-special-1-2023', 'frieren-beyond-journeys-end', 'tapping-the-wire', ]
