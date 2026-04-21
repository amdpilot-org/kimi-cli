from importlib import import_module


def test_python310_datetime_timezone_imports() -> None:
    modules = (
        "kimi_cli.web.store.sessions",
        "kimi_cli.web.runner.process",
        "kimi_cli.ui.shell.usage",
        "kimi_cli.web.api.sessions",
    )

    for module_name in modules:
        import_module(module_name)
