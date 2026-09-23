"""
Copyright (C) 2026 Laurent Courty

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from __future__ import annotations

import json

import itzi.messenger as msgr
from itzi.cloud import urls
from itzi.cloud.schemas import ProjectSchema

try:
    import requests
except ImportError:
    raise ImportError(
        "To use the cloud functionalities, install itzi with: "
        "'uv tool install itzi[cloud]' "
        "or 'pip install itzi[cloud]'"
    )


def get_projects_list(session_token: str, url: str | None = None) -> list[ProjectSchema]:
    """Retrieve active projects belonging to the authenticated user's teams."""
    url = url or urls.get_projects_endpoint()
    headers = {"X-Session-Token": session_token}

    with requests.Session() as session:
        response = session.get(url, headers=headers)

        if response.status_code != 200:
            msgr.fatal(
                f"Failed to retrieve projects. "
                f"Code: {response.status_code}. Reason: {response.reason}"
            )

        response_data = json.loads(response.text)

    return [ProjectSchema.model_validate(project_data) for project_data in response_data]


def display_projects_list(projects: list[ProjectSchema]) -> None:
    """Display projects and their teams in a table."""
    if not projects:
        msgr.message("No projects found.")
        return

    header = f"{'PROJECT':<25} {'PROJECT SLUG':<25} {'TEAM':<25} {'TEAM SLUG':<25}"
    msgr.message(header)

    for project in projects:
        row = (
            f"{project.name:<25} {project.slug:<25} "
            f"{project.team.name:<25} {project.team.slug:<25}"
        )
        msgr.message(row)
