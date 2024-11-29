import datetime

def get_date_string() -> str:
    return datetime.datetime.now().isoformat()[0:19].replace("-", "").replace(":", "").replace(".", "")

