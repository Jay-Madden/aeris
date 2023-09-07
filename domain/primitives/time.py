import datetime
from typing import Annotated

import pytz

from llm.session import SessionGroup, Param

group = SessionGroup()


@group.function("Get the current time.py in UTC ISO-8601 format")
def get_current_time(
    tz_name: Annotated[
        str,
        Param(
            description="Olsen tz name of the timezone to get the current time.py of"
        ),
    ]
) -> str:
    tz = pytz.timezone(tz_name)
    return tz.fromutc(datetime.datetime.utcnow()).isoformat()
