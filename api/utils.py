import math
from typing import Any, Dict, List

from fastapi import Request


def get_user_id(request: Request) -> str:
    return request.headers.get("user-id", request.client.host)


def clean_and_remove_empty_columns(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Efficiently remove columns that are entirely None/NaN.
    Single pass to identify valid columns, single pass to clean.
    """
    if not data_list:
        return data_list

    valid_columns = set()
    columns_to_check = set(data_list[0].keys())

    for row in data_list:
        cols_to_remove = set()
        for col in columns_to_check:
            if col in row:
                value = row[col]
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    valid_columns.add(col)
                    cols_to_remove.add(col)

        columns_to_check -= cols_to_remove

        if not columns_to_check:
            break

    cleaned_data = []
    for row in data_list:
        cleaned_row = {}
        for col in valid_columns:
            if col in row:
                value = row[col]
                if isinstance(value, float) and math.isnan(value):
                    cleaned_row[col] = None
                else:
                    cleaned_row[col] = value
        cleaned_data.append(cleaned_row)

    return cleaned_data
