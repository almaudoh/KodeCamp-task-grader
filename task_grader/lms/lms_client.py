import datetime
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import requests


class SubmissionCategory(StrEnum):
    GRADED = "graded"
    SUBMITTED = "submitted"
    ALL = "all"


@dataclass
class SubmissionMeta:
    task_id: str
    submission_id: str
    trainee_id: str
    trainee_name: str
    submission_date: str
    due_date: str
    solution_urls: list[str]
    submission_status: str
    score: float = 0.0


class LMSClient:
    def __init__(self, base_url: str, email: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self._token: dict[str, str] | None = None
        self._session = requests.Session()

    def login(self) -> None:
        """
        Authenticate with LMS and return the bearer token.
        Expects /login to take JSON with:
            {
              "grant_type": "password",
              "username": "...",
              "password": "...",
            }
        And respond with a JSON that includes an access token and its expiration date.:
            { "access_token": "...", "expires": "..." }
        """
        login_url = f"{self.base_url}/login"
        payload = {
            "grant_type": "password",
            "username": self.email,
            "password": self.password,
        }

        resp = self._session.post(login_url, data=payload)

        if not resp.ok:
            raise RuntimeError(
                f"Login failed with status {resp.status_code}.\nResponse text:\n{resp.text}"
            )

        data = resp.json()
        token_string = data.get("access_token")
        token_expires = data.get("expires")

        # now = datetime.datetime.now(tz=datetime.timezone.utc)
        now = datetime.datetime.now()

        if (
            not token_string
            or not token_expires
            or datetime.datetime.fromisoformat(token_expires) <= now
        ):
            raise RuntimeError("No valid access_token in login response.")

        self._token = {
            "token_string": token_string,
            "token_expires": token_expires,
        }

        self._session.headers.update({"Authorization": f"Bearer {token_string}"})

    def logout(self) -> None:
        logout_url = f"{self.base_url}/logout"
        resp = self._session.delete(logout_url)
        if resp.status_code not in (200, 204):
            # Don't crash if logout fails
            print(f"[WARN] Logout failed with status {resp.status_code}: {resp.text}")

        # Clear the access token on successful logout
        self._token = None
        del self._session.headers["Authorization"]

    def is_token_valid(self) -> bool:
        if not self._token or not self._session.headers.get("Authorization"):
            return False

        token_validation_url = f"{self.base_url}/test-access-token"
        resp = self._session.post(token_validation_url)

        if not resp.ok:
            return False

        data = resp.json()

        if not (data.get("email") == self.email):
            return False

        return True

    def get_task_submissions(
        self,
        task_id: str,
        workspace_slug: str,
        category: SubmissionCategory = SubmissionCategory.ALL,
        offset: int = 0,
        limit: int = 100,
    ) -> list[SubmissionMeta]:
        """
        Fetch all submissions for a given task.
        """
        if not self.is_token_valid():
            self.login()

        submissions_retrieval_url = f"{self.base_url}/{workspace_slug}/submissions"
        params = {
            "offset": offset,
            "limit": limit,
        }
        resp = self._session.get(submissions_retrieval_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if (
            not isinstance(data, dict)
            or not data.get("submissions")
            or not data.get("students")
        ):
            raise RuntimeError("Could not retrieve the expected task submissions data")

        if not isinstance(data.get("submissions"), list) or not isinstance(
            data.get("students"), list
        ):
            raise RuntimeError(
                "Retrieved submissions data does not match the expected format"
            )

        submissions: list[SubmissionMeta] = []

        for item in data["submissions"]:
            trainee_id = item["student_id"]
            trainee_profile = [
                profile["profile"]
                for profile in data["students"]
                if profile["id"] == trainee_id
            ][0]
            submission = SubmissionMeta(
                task_id=task_id,
                submission_id=str(item["id"]),
                trainee_id=str(trainee_id),
                trainee_name=f"{trainee_profile.get('first_name')} {trainee_profile.get('last_name')}".strip(),
                submission_date=str(item["updated_at"]),
                due_date=str(item["task"]["due_date"]),
                solution_urls=item["submission_urls"],
                submission_status=item["status"],
                score=item["score"],
            )

            if (
                category == SubmissionCategory.ALL
                or category == submission.submission_status
            ):
                submissions.append(submission)

        return submissions

    def get_task_with_submissions(
        self,
        task_id: str,
        workspace_slug: str,
    ) -> dict:
        """
        Fetch task details along with its submissions.
        """
        if not self.is_token_valid():
            self.login()

        task_retrieval_url = f"{self.base_url}/{workspace_slug}/tasks/{task_id}"
        resp = self._session.get(task_retrieval_url)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict) or not data.get("task"):
            raise RuntimeError("Could not retrieve the expected task data")

        return data

    def get_token(self):
        return self._token

    def get_session(self):
        return self._session

    @classmethod
    def from_env(cls) -> Self:
        base_url = os.getenv("LMS_BASE_URL")
        username = os.getenv("LMS_EMAIL")
        password = os.getenv("LMS_PASSWORD")

        if not base_url or not username or not password:
            raise RuntimeError(
                "At least one of the environmental, `LMS_BASE_URL`, `LMS_EMAIL`, and `LMS_PASSWORD`, is not set"
            )

        return cls(base_url, username, password)
