"""get_datetime: the current date and time here or elsewhere, and days until a date."""

from __future__ import annotations

import re
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from ai_buddy.tools import Tool, ToolContext, ToolResult

# Common places without a zone of their own.
ALIASES = {
    "calgary": "America/Edmonton", "alberta": "America/Edmonton", "ottawa": "America/Toronto",
    "montreal": "America/Toronto", "quebec": "America/Toronto", "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles", "silicon valley": "America/Los_Angeles", "boston": "America/New_York",
    "washington": "America/New_York", "miami": "America/New_York", "atlanta": "America/New_York",
    "dallas": "America/Chicago", "houston": "America/Chicago", "austin": "America/Chicago",
    "beijing": "Asia/Shanghai", "china": "Asia/Shanghai", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata", "india": "Asia/Kolkata", "japan": "Asia/Tokyo", "osaka": "Asia/Tokyo",
    "uk": "Europe/London", "england": "Europe/London", "france": "Europe/Paris", "germany": "Europe/Berlin",
    "munich": "Europe/Berlin", "spain": "Europe/Madrid", "barcelona": "Europe/Madrid", "italy": "Europe/Rome",
    "milan": "Europe/Rome", "netherlands": "Europe/Amsterdam", "korea": "Asia/Seoul", "seoul": "Asia/Seoul",
    "hawaii": "Pacific/Honolulu", "utc": "UTC", "gmt": "UTC",
}


def resolve_zone(name: str) -> ZoneInfo | None:
    name = name.strip()
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        pass
    key = name.lower().replace("_", " ")
    if key in ALIASES:
        return ZoneInfo(ALIASES[key])
    wanted = "/" + key.replace(" ", "_")
    for zone in sorted(available_timezones()):
        if zone.lower().endswith(wanted):
            return ZoneInfo(zone)
    return None


# "what time is it", "what's the time", "the current time"; not "what time is the meeting"
_TIME_Q = re.compile(r"\bwhat(?:'s| is)? the (?:current |local )?time\b(?! (?:of|for|on|at|when|difference|zone)\b)"
                     r"|\bwhat time(?: is it|'s it| it is)\b|\btime is it\b|\bcurrent time\b(?! (?:difference|zone)\b)",
                     re.I)
_PLACE = re.compile(r"\b(?:in|for|at|to|from|visiting)\s+(?:the\s+)?(?P<place>[A-Za-z][\w'.-]*(?:\s+[A-Za-z][\w'.-]*){0,2})")
_UNTIL = re.compile(r"\bhow (?:many|much) (?:more )?(?:days|weeks|time)\b[^?.!]*?\b(?:until|till|til|before|to)\s+"
                    r"(?P<what>[^?.!,]+)", re.I)
HOLIDAYS = {"christmas eve": (12, 24), "christmas": (12, 25), "boxing day": (12, 26), "new year's eve": (12, 31),
            "new years eve": (12, 31), "new year's": (1, 1), "new years": (1, 1), "new year": (1, 1),
            "halloween": (10, 31), "valentine's day": (2, 14), "valentines day": (2, 14), "canada day": (7, 1),
            "independence day": (7, 4)}
MONTHS = {m: i for i, m in enumerate("january february march april may june july august september october november "
                                     "december".split(), 1)}


def place_zone(text: str) -> ZoneInfo | None:
    """The zone of a place named in `text` ("in New York right now" -> America/New_York), trying the longest
    phrase first."""
    for m in _PLACE.finditer(text):
        words = m["place"].split()
        for n in range(len(words), 0, -1):
            if zone := resolve_zone(" ".join(words[:n])):
                return zone
    return None


def target_date(what: str, today: date) -> date | None:
    """'Christmas' -> the next Dec 25; 'October 30th' -> that date (next year if it has passed)."""
    key = what.strip().lower().removeprefix("the ").strip()
    month_day = HOLIDAYS.get(key)
    if month_day is None:
        if m := re.match(r"(?P<m>[a-z]+)\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?\b", key):
            month_day = (MONTHS[m["m"]], int(m["d"])) if m["m"] in MONTHS else None
    if month_day is None:
        return None
    try:
        d = date(today.year, *month_day)
        return d if d >= today else date(today.year + 1, *month_day)
    except ValueError:
        return None


def describe_time(now: datetime) -> dict:
    return {"date": now.strftime("%A, %B %-d, %Y"), "time": now.strftime("%-I:%M %p"),
            "iso": now.isoformat(timespec="minutes"), "timezone": str(now.tzinfo)}


class DateTime(Tool):
    name = "get_datetime"
    description = "Get the current date and time, locally or in another timezone, and optionally the days until a date."
    parameters = {
        "timezone": {"type": "string",
                     "description": "IANA timezone such as 'Asia/Tokyo' (a city name also works). Omit for local time."},
        "until": {"type": "string", "description": "A date as YYYY-MM-DD to count the days until (optional)."},
    }
    required = []
    label = "Checking the time"
    filler = "One sec."

    def describe(self, args: dict) -> str:
        return args.get("timezone") or ""

    def route(self, text: str, ctx: ToolContext) -> dict | None:
        if m := _UNTIL.search(text):
            target = target_date(m["what"], ctx.today())
            return {"until": target.isoformat()} if target else None
        if not (q := _TIME_Q.search(text)):
            return None
        after = text[q.end():]
        if _PLACE.search(after):
            zone = place_zone(after)
            return {"timezone": str(zone)} if zone else None   # a place we can't place: let the model ask
        if re.search(r"\bthere\b", after, re.I):
            zone = place_zone(text[:q.start()])     # "the weather in Paris, and what time is it there?"
            return {"timezone": str(zone)} if zone else None
        return {}

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        local = ZoneInfo(ctx.timezone)
        out = {"local": describe_time(datetime.now(local))}
        summary = out["local"]["time"]
        if tz := str(args.get("timezone") or "").strip():
            zone = resolve_zone(tz)
            if zone is None:
                out["error"] = f"Unknown timezone {tz!r}; use an IANA name like 'Europe/Paris'."
            else:
                out["requested"] = describe_time(datetime.now(zone))
                summary = f"{out['requested']['time']} in {zone}"
        if until := str(args.get("until") or "").strip():
            try:
                target = date.fromisoformat(until[:10])
                out["days_until"] = {"date": target.isoformat(), "days": (target - datetime.now(local).date()).days}
                summary = f"{out['days_until']['days']} days until {target.isoformat()}"
            except ValueError:
                out["error"] = f"Could not read the date {until!r}; use YYYY-MM-DD."
        return ToolResult(out, summary=summary, ok="error" not in out)
