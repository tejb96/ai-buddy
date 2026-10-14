"""get_datetime: the current date and time here or elsewhere, and days until a date."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from . import Tool, ToolContext, ToolResult

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
