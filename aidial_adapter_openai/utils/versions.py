# Azure API versions follow the next formats: YYYY-MM-DD or YYYY-MM-DD-preview
def compare_versions(v1: str, v2: str):
    if len(v1) < 10 or len(v2) < 10:
        return None

    v1 = v1[0:10]
    v2 = v2[0:10]

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0
