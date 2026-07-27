import nox

nox.options.reuse_existing_virtualenvs = True

SRC = ["aidial_adapter_openai", "tests", "noxfile.py"]


@nox.session
def lint(session: nox.Session):
    """Runs linters and fixers"""
    try:
        session.run("poetry", "install", "--with", "lint", external=True)
        session.run("poetry", "check", "--lock", "--strict", external=True)
        session.run("ruff", "check", *SRC)
        session.run("ruff", "format", "--check", *SRC)
        session.run("pyright", *SRC)
    except Exception:
        session.error(
            "linting has failed. Run 'make format' to fix formatting and fix other errors manually"
        )


@nox.session
def format(session: nox.Session):
    """Runs linters and fixers"""
    session.run("poetry", "install", "--only", "lint", external=True)
    session.run("ruff", "check", "--fix", *SRC)
    session.run("ruff", "format", *SRC)


@nox.session
def test(session: nox.Session):
    """Runs unit tests"""
    session.run("poetry", "install", external=True)
    session.run("pytest", "tests/unit_tests")


@nox.session
def integration_test(session: nox.Session):
    """Runs integration tests"""
    session.run("poetry", "install", external=True)
    session.run("pytest", "tests/integration_tests")
