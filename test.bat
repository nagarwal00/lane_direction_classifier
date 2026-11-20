@echo off
"%~dp0.venv\Scripts\pytest.exe" "%~dp0tests\test_pipeline.py" %*
