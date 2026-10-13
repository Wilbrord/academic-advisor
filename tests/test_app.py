import pytest

from app import app


@pytest.fixture
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Academic standing" in response.data


def test_assessment_returns_prediction_and_support(client):
    response = client.post(
        "/",
        data={
            "gpa": "2.1",
            "attendance_pct": "65",
            "weekly_study_hours": "6",
            "credits_enrolled": "19",
            "prior_failed_courses": "2",
        },
    )

    assert response.status_code == 200
    assert b"Estimated risk" in response.data
    assert b"Book time with an academic advisor" in response.data


def test_assessment_rejects_out_of_range_values(client):
    response = client.post(
        "/",
        data={
            "gpa": "5",
            "attendance_pct": "85",
            "weekly_study_hours": "12",
            "credits_enrolled": "15",
            "prior_failed_courses": "0",
        },
    )

    assert response.status_code == 200
    assert b"Enter valid values" in response.data


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}
