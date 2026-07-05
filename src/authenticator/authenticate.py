import ee
def authenticateuser(project_name):
    try:
        ee.Initialize(project=project_name)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_name)

